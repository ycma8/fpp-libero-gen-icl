"""Task-safe support/query pairing; test-task videos never enter training."""
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from fpp_libero_mem.data import action_chunks, normalize
from fpp_libero_gen.sensorimotor import SensorimotorCache
from fpp_libero_gen.execution import ExecutionCache

SPLITS = {
    'combination': {'train': ['libero_spatial_selected_combinations_inverse_view'],
                    'test': ['libero_spatial_selected_combinations_view']},
    'chain': {'train': ['libero_goal_chain_firststep_view', 'libero_goal_chain_secondstep_view',
                        'libero_goal_chain_selected_inverse_view'],
              'test': ['libero_goal_chain_selected_view']},
}


def sample_demo(features, frames):
    if len(features) < 1:
        raise ValueError('Empty demo')
    indices = np.linspace(0, len(features)-1, frames).round().astype(int)
    return np.asarray(features[indices]).copy()


class PairedEpisodes(torch.utils.data.Dataset):
    def __init__(self, root, split='train', demo_frames=32, target_steps=16, demo_cache_root=None, execution_cache_root=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.episodes = [e for e in self.manifest['episodes'] if e['split'] == split]
        self.by_task = defaultdict(list)
        for i, e in enumerate(self.episodes):
            self.by_task[e['task']].append(i)
        if not self.episodes or any(len(v) < 2 for v in self.by_task.values()):
            raise ValueError('Each split/task needs at least two distinct episodes')
        self.demo_frames, self.target_steps = demo_frames, target_steps
        self.demo_cache = SensorimotorCache(demo_cache_root,
            hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest()) if demo_cache_root else None
        self.execution_cache = ExecutionCache(execution_cache_root,
            hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest(),
            self.demo_cache.sha256 if self.demo_cache else None) if execution_cache_root else None
        if self.execution_cache and not self.demo_cache:
            raise ValueError('BPP execution input requires its demo modalities')

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        ep = self.episodes[index]
        candidates = [i for i in self.by_task[ep['task']] if i != index]
        support = self.episodes[int(np.random.choice(candidates))]
        features = np.load(self.root/ep['cache']/'features.npy', mmap_mode='r')
        actions = None if self.execution_cache else np.load(self.root/ep['cache']/'actions.npy', mmap_mode='r')
        if self.demo_cache is None:
            support_features = np.load(self.root/support['cache']/support.get('demo_feature_file', 'features.npy'), mmap_mode='r')
        indices = np.load(self.root/ep['cache']/'indices.npy')
        extra = {}
        stats = self.manifest['normalization']
        if self.execution_cache:
            wrist, proprio, actions = self.execution_cache.load(ep['id'], indices)
            features = np.stack((features, wrist), axis=1)
            extra['query_proprio'] = torch.from_numpy(proprio)
            stats = self.execution_cache.manifest['normalization']['action']
        if len(indices) > self.manifest['max_steps']:
            raise ValueError('Episode exceeds history capacity')
        valid = np.zeros(len(indices), dtype=bool)
        valid[np.random.choice(len(indices), min(len(indices), self.target_steps), replace=False)] = True
        targets = action_chunks(actions, indices, self.manifest['horizon'])
        prompt = self.demo_cache.load(support['id']) if self.demo_cache is not None else dict(
            demo=sample_demo(support_features, self.demo_frames))
        return dict(features=torch.from_numpy(np.asarray(features).copy()),
            **{k:torch.from_numpy(v) for k,v in prompt.items()}, **extra,
            actions=torch.from_numpy(normalize(targets, stats).astype('float32')),
            valid=torch.from_numpy(valid), query_id=ep['id'], support_id=support['id'], task=ep['task'])


def collate(samples, max_steps=None):
    length = max_steps or max(len(s['features']) for s in samples)
    if any(len(s['features']) > length for s in samples):
        raise ValueError('Query exceeds fixed context capacity')
    result = {key:torch.stack([s[key] for s in samples]) for key in
              ['demo','demo_proprio','demo_actions','demo_valid'] if key in samples[0]}
    for key in ['features', 'actions', 'valid'] + (['query_proprio'] if 'query_proprio' in samples[0] else []):
        result[key] = torch.zeros((len(samples), length, *samples[0][key].shape[1:]), dtype=samples[0][key].dtype)
        for i, sample in enumerate(samples):
            result[key][i, :len(sample[key])] = sample[key]
    return result
