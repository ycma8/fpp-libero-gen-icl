"""BPP-compatible demo fields/chunks, independent of the execution observations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fpp_libero_mem.patch_causal_dit import sinusoidal


def rotation_6d(axis_angle):
    """The first two matrix rows, matching BPP/PyTorch3D rotation_6d."""
    a = np.asarray(axis_angle, dtype=np.float64)
    theta = np.linalg.norm(a, axis=-1, keepdims=True)
    xyz = a * (.5 * np.sinc(theta / (2 * np.pi)))
    w = np.cos(theta / 2)[..., 0]
    x, y, z = np.moveaxis(xyz, -1, 0)
    result = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                       2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)), -1)
    return result.astype(np.float32)


def convert_fields(actions, position, orientation, gripper):
    actions = np.asarray(actions, dtype=np.float32)
    proprio = np.concatenate((position, rotation_6d(orientation), gripper), -1).astype(np.float32)
    action = np.concatenate((actions[:, :3], rotation_6d(actions[:, 3:6]), actions[:, 6:]), -1)
    if action.shape != (len(actions), 10) or proprio.shape != (len(actions), 11):
        raise ValueError('Expected BPP action10/proprio11 fields at every source timestep')
    if not np.isfinite(action).all() or not np.isfinite(proprio).all():
        raise ValueError('Nonfinite sensorimotor demo')
    return action, proprio


def chunk_fields(actions, proprio, chunk_size=20):
    """BPP pairs obs[j] with source actions[j:j+20], not query's offset+1."""
    length = len(actions)
    if length < 1 or len(proprio) != length or chunk_size < 1:
        raise ValueError('Invalid demo length/chunk size')
    indices = np.arange(0, length, chunk_size)
    padded = np.zeros((len(indices)*chunk_size, actions.shape[-1]), dtype=np.float32)
    padded[:length] = actions
    return indices, padded.reshape(len(indices), chunk_size, -1), proprio[indices].copy()


def range_spec(low, high, ranged):
    low, high = np.asarray(low), np.asarray(high)
    center, scale = np.zeros_like(low), np.ones_like(low)
    selected = np.asarray(ranged, dtype=int)
    center[selected] = (low[selected]+high[selected])/2
    width = high[selected]-low[selected]
    scale[selected] = np.where(width < 1e-8, 1, width/2)
    return dict(center=center.tolist(), scale=scale.tolist())


class SensorimotorCache:
    def __init__(self, root, base_manifest_sha256=None):
        self.root = Path(root)
        raw = (self.root/'manifest.json').read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest['schema'] != 'fpp_bpp_demo_v1':
            raise ValueError('Unknown sensorimotor demo cache')
        if base_manifest_sha256 and self.manifest['base_manifest_sha256'] != base_manifest_sha256:
            raise ValueError('Demo/query task/data identities differ')
        self.records = {e['id']: e for e in self.manifest['episodes']}
        if len(self.records) != len(self.manifest['episodes']):
            raise ValueError('Duplicate demo identities')

    def load(self, episode_id):
        ep = self.records[episode_id]
        p = self.root/ep['cache']
        features = np.load(p/'demo_features.npy')
        with np.load(p/'sensorimotor.npz') as low:
            actions, proprio = low['actions'], low['proprio']
        n = len(features)
        maximum = self.manifest['max_chunks']
        if n != ep['chunks'] or not 0 < n <= maximum:
            raise ValueError('Demo exceeds configured capacity')
        result = dict(demo=np.zeros((maximum, *features.shape[1:]), dtype=features.dtype),
            demo_actions=np.zeros((maximum, *actions.shape[1:]), dtype=np.float32),
            demo_proprio=np.zeros((maximum, 11), dtype=np.float32),
            demo_valid=np.arange(maximum) < n)
        result['demo'][:n] = features
        for key, data, spec in [('demo_actions', actions, 'action'), ('demo_proprio', proprio, 'proprio')]:
            norm = self.manifest['normalization'][spec]
            result[key][:n] = (data-np.asarray(norm['center']))/np.asarray(norm['scale'])
        return result


def transform_prompt(prompt, condition, seed):
    """Keep all modalities synchronized; padding remains invisible to attention."""
    result = {k: v.copy() for k,v in prompt.items()}
    n = int(result['demo_valid'].sum())
    order = np.arange(n)
    if condition == 'shuffled':
        order = np.random.default_rng(seed+98173).permutation(n)
    elif condition == 'last_frame':
        order = np.full(n, n-1)
    elif condition not in ('correct', 'wrong', 'blank'):
        raise ValueError('Unknown prompt control')
    for key in ['demo', 'demo_actions', 'demo_proprio']:
        result[key][:n] = result[key][order]
        if condition == 'blank':
            result[key].fill(0)
    return result, order.tolist()


class SensorimotorEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config['visual_dim']
        self.grid = config['grid_size']
        self.full_patches = config.get('demo_full_patches', False)
        self.demo_grid = self.grid if self.full_patches else config['demo_grid_size']
        self.chunk_size = config['demo_chunk_size']
        self.view = nn.Parameter(torch.randn(2, dim)*.02)
        self.modality = nn.Parameter(torch.randn(2, dim)*.02)
        self.proprio = nn.Sequential(nn.Linear(11, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action = nn.Sequential(nn.Linear(10, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer('action_position', sinusoidal(torch.arange(self.chunk_size), dim))

    @property
    def tokens_per_chunk(self):
        return 2*self.demo_grid**2 + 1 + self.chunk_size

    def forward(self, demo, proprio, actions, norm, projection, spatial, segment, image_pool=None):
        b,t,v,p,d = demo.shape
        if v != 2 or p != self.grid**2 or actions.shape != (b,t,self.chunk_size,10) or proprio.shape != (b,t,11):
            raise ValueError('BPP demo layout differs from config')
        if image_pool is None:
            if self.full_patches:
                pooled = demo
            else:
                pooled = F.adaptive_avg_pool2d(demo.reshape(b*t*v,self.grid,self.grid,d).permute(0,3,1,2),
                                              (self.demo_grid,self.demo_grid))
                pooled = pooled.permute(0,2,3,1).reshape(b,t,v,self.demo_grid**2,d)
            image = projection(norm(pooled))
            image = image + spatial.to(image.dtype)[None,None,None] + self.view.to(image.dtype)[None,None,:,None]
        else:
            if self.full_patches:
                raise ValueError('Full demo patches cannot use an image pooling function')
            image = image_pool(demo, self.view)
        proprio_token = self.proprio(proprio) + self.modality[0].to(image.dtype)
        action_tokens = self.action(actions) + self.modality[1].to(image.dtype) + self.action_position.to(image.dtype)[None,None]
        tokens = torch.cat((image.flatten(2,3), proprio_token[:,:,None], action_tokens), 2)
        return (tokens + segment.to(tokens.dtype)).flatten(1,2)
