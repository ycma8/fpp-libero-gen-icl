"""Audited dual-view/proprio execution sidecar; no action labels in observations."""
import hashlib
import json
from pathlib import Path

import numpy as np


class ExecutionCache:
    def __init__(self, root, base_sha=None, demo_sha=None):
        self.root = Path(root)
        raw = (self.root/'manifest.json').read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest['schema'] != 'fpp_execution_bpp_v1':
            raise ValueError('Unknown execution cache')
        if base_sha and self.manifest['base_manifest_sha256'] != base_sha:
            raise ValueError('Execution/base identity differs')
        if demo_sha and self.manifest['demo_manifest_sha256'] != demo_sha:
            raise ValueError('Execution/demo identity differs')
        self.records = {e['id']: e for e in self.manifest['episodes']}
        if len(self.records) != len(self.manifest['episodes']):
            raise ValueError('Duplicate execution IDs')

    def load(self, episode_id, indices):
        ep = self.records[episode_id]
        path = self.root/ep['cache']
        wrist = np.load(path/'wrist_features.npy', mmap_mode='r')
        with np.load(path/'lowdim.npz') as low:
            if not np.array_equal(indices, low['indices']):
                raise ValueError('Query camera/proprio timestamps differ')
            actions, proprio = low['actions'].copy(), low['proprio'].copy()
        if len(wrist) != len(indices) or proprio.shape != (len(indices), 11):
            raise ValueError('Invalid execution layout')
        return wrist, self.normalize_proprio(proprio), actions

    def normalize_proprio(self, value):
        spec = self.manifest['normalization']['proprio']
        return ((value-np.asarray(spec['center']))/np.asarray(spec['scale'])).astype(np.float32)


def policy_proprio(obs):
    """Match official prepare_obs_dict: xyzw -> wxyz -> rotation6d."""
    from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
    xyzw = np.asarray(obs['robot0_eef_quat'][-1], dtype=np.float32)
    q = RotationTransformer(from_rep='quaternion', to_rep='matrix')
    matrix = q.forward(xyzw[[3, 0, 1, 2]])
    ori = matrix[:2].reshape(6)
    return np.concatenate((obs['robot0_eef_pos'][-1], ori, obs['robot0_gripper_qpos'][-1])).astype(np.float32)


def native_action(action):
    from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != 10:
        raise ValueError('Expected10D action')
    rotation = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')
    return np.concatenate((action[..., :3], rotation.inverse(action[..., 3:9]), action[..., 9:]), -1)
