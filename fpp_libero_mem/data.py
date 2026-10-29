"""Episode-safe feature and action data for LIBERO-Mem."""
from pathlib import Path
import hashlib
import json
import os
import re

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get('FPP_DATA_ROOT', 'data/libero-mem'))
CAMERAS = ['agentview_rgb']
CAMERA_MODES = {
    'agentview': (0,),
}
TASK_MIN_STEPS = [200, 290, 320, 370, 450, 655, 480, 675, 390, 390]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def task_number(path):
    return int(re.search(r'SCENE1_(\d+)_', str(path)).group(1))


def text_for(path):
    return re.sub(r'^KITCHEN_SCENE1_\d+_', '', Path(path).name.removesuffix('_demo.hdf5')).replace('_', ' ')


def strip_leading(actions):
    nonzero = np.flatnonzero(np.linalg.norm(actions[:, :-1], axis=1) >= 1e-4)
    if not len(nonzero):
        raise ValueError('All-noop demonstration')
    return actions[nonzero[0]:]


def pad_resize(images):
    """BCHW uint8 -> BCHW uint8; identical conversion offline and during rollout."""
    h, w = images.shape[-2:]
    size = max(h, w)
    images = F.pad(images, ((size-w)//2, size-w-(size-w)//2,
                            (size-h)//2, size-h-(size-h)//2))
    return F.interpolate(images.float(), (224, 224), mode='bilinear',
                         align_corners=False, antialias=True).round().clamp(0, 255).to(torch.uint8)


def audit_raw(root=ROOT):
    tasks = []
    for path in sorted((root/'raw').glob('*.hdf5'), key=task_number):
        with h5py.File(path) as f:
            group = f['data']
            keys = sorted(group, key=lambda x: int(x.split('_')[-1]))
            if len(keys) < 40:
                raise ValueError(f'Too few raw episodes: {path}')
            episodes = []
            for key in keys:
                a = group[key]['actions'][()]
                initial = group[key]['init_state'][()]
                if a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all():
                    raise ValueError(f'Invalid native action data: {path}/{key}')
                episodes.append(dict(id=key, length=len(a),
                    horizon=max(TASK_MIN_STEPS[task_number(path)-1], len(strip_leading(a))+20),
                    init_sha=hashlib.sha256(initial.astype(np.float64).tobytes()).hexdigest()))
            train_hashes = {e['init_sha'] for e in episodes[:-20]}
            test_hashes = {e['init_sha'] for e in episodes[-20:]}
            if train_hashes & test_hashes:
                raise ValueError(f'Train/test initial-state duplication: {path.name}')
            tasks.append(dict(number=task_number(path), file=path.name, text=text_for(path),
                              episodes=episodes, test_ids=keys[-20:]))
    if len(tasks) != 10:
        raise ValueError(f'Expected all ten raw tasks, found {len(tasks)}')
    result = dict(tasks=tasks,
                  max_raw_length=max(e['length'] for t in tasks for e in t['episodes']),
                  max_eval_horizon=max(e['horizon'] for t in tasks for e in t['episodes']))
    dump(root/'prepared/raw_audit.json', result)
    return result


def action_chunks(actions, indices, horizon=16):
    # Labels NEVER cross an episode boundary; endpoint holds are explicit.
    offsets = np.minimum(indices[:, None] + np.arange(horizon), len(actions)-1)
    return actions[offsets]


def normalize(actions, stats):
    return (actions - np.asarray(stats['center'])) / np.asarray(stats['scale'])


def denormalize(actions, stats):
    result = actions * np.asarray(stats['scale']) + np.asarray(stats['center'])
    result[..., np.asarray(stats['constant'], dtype=bool)] = np.asarray(stats['center'])[stats['constant']]
    return result


class Episodes(torch.utils.data.Dataset):
    def __init__(self, root, split='train', mode='full', camera_mode='agentview',
                 append_eoe=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'prepared/manifest.json').read_text())
        self.stats = self.manifest['normalization']
        self.episodes = [e for e in self.manifest['episodes'] if e['split'] == split]
        if mode != 'full':
            raise ValueError('This dataset supports FPP feature caches only')
        if self.manifest.get('camera_keys') != CAMERAS:
            raise ValueError('Expected single-agentview cache; rebuild it with prepare_cache.py')
        self.mode = mode
        if camera_mode not in CAMERA_MODES:
            raise ValueError(f'Unknown camera mode: {camera_mode}')
        self.camera_mode = camera_mode
        self.camera_indices = CAMERA_MODES[camera_mode]
        self.text = np.load(self.root/'prepared/text.npy')
        self.stride = self.manifest['history_stride']
        self.append_eoe = append_eoe

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        ep = self.episodes[index]
        directory = self.root/'prepared'/ep['cache']
        actions = np.load(directory/'actions.npy', mmap_mode='r')
        text = torch.from_numpy(self.text[ep['task']-1].copy())
        phase = np.random.randint(min(self.stride, len(actions)))
        indices = np.arange(phase, len(actions), self.stride)
        feature_cache = np.load(directory/'features.npy', mmap_mode='r')
        features = feature_cache[indices].copy()
        features = features.reshape(len(indices), len(CAMERAS), -1, features.shape[-1])
        features = features[:, self.camera_indices].reshape(len(indices), -1, features.shape[-1])
        targets = action_chunks(actions, indices)
        if self.append_eoe:
            # Reference recipe: one extra terminal observation/hold target.
            last_features = feature_cache[-1:].copy()
            features = np.concatenate((features, last_features))
            targets = np.concatenate((targets, np.repeat(actions[-1:][None], 16, axis=1)))
        if len(features) > self.manifest['max_observations']:
            raise ValueError('Episode exceeds max_observations; increase capacity or history_stride')
        return dict(features=torch.from_numpy(features), text=text,
                    actions=torch.from_numpy(normalize(targets, self.stats).astype('float32')),
                    valid=torch.ones(len(features), dtype=torch.bool))


def full_collate(samples):
    length = max(len(s['features']) for s in samples)
    out = {}
    for key in ['features', 'actions', 'valid']:
        shape = (len(samples), length, *samples[0][key].shape[1:])
        out[key] = torch.zeros(shape, dtype=samples[0][key].dtype)
        for i, sample in enumerate(samples):
            out[key][i, :len(sample[key])] = sample[key]
    out['text'] = torch.stack([s['text'] for s in samples])
    return out
