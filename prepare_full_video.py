"""Cache every training demo; score official pruned initial states, not demo states.

Run disjoint task shards, then --merge. Uses the already SHA-pinned official
source manifest from the pilot; no task or episode subsampling is permitted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import timm
import torch

from fpp_libero_gen.data import SPLITS
from prepare_video import digest
from scripts.prepare_cache import _features


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source_manifest.read_text())
    assert source['suite'] == 'combination'
    tasks = source['tasks']
    train_tasks = {x['task'] for x in tasks if x['view'] in SPLITS['combination']['train']}
    test_tasks = {x['task'] for x in tasks if x['view'] in SPLITS['combination']['test']}
    assert len(train_tasks) == 164 and len(test_tasks) == 10 and not train_tasks & test_tasks
    identity = dict(source_manifest_sha256=digest(args.source_manifest),
                    dino_checkpoint_sha256=source['dino_checkpoint_sha256'],
                    history_stride=args.history_stride, action_offset=1,
                    max_steps=args.max_steps, task_shards=args.shards)
    if args.merge_shard:
        parts = [json.loads((args.output/f'shard_{args.shard}.sub_{i}.json').read_text())
                 for i in range(args.subshards)]
        assert all(p['identity'] == identity for p in parts)
        records = [r for part in parts for r in part['episodes']]
        records.sort(key=lambda r: (r['view'], r['task'], int(r['episode'].split('_')[-1])))
        assert len({r['id'] for r in records}) == len(records)
        part = dict(identity=identity, episodes=records,
                    low=np.min([p['low'] for p in parts], axis=0).tolist(),
                    high=np.max([p['high'] for p in parts], axis=0).tolist())
        (args.output/f'shard_{args.shard}.json').write_text(json.dumps(part, indent=2))
        return
    if args.merge:
        parts = [json.loads((args.output/f'shard_{i}.json').read_text()) for i in range(args.shards)]
        assert all(p['identity'] == identity for p in parts)
        records = [r for part in parts for r in part['episodes']]
        records.sort(key=lambda r: (r['view'], r['task'], int(r['episode'].split('_')[-1])))
        assert len({r['id'] for r in records}) == len(records)
        assert {r['task'] for r in records} == train_tasks | test_tasks
        assert sum(r['split'] == 'train' for r in records) == 8200
        assert sum(r['split'] == 'test_support' for r in records) == 500
        for record in records:
            directory = args.output/record['cache']
            demo = json.loads((directory/'demo_record.json').read_text())
            assert demo['checkpoint_sha256'] == identity['dino_checkpoint_sha256']
            assert demo['source_sha256'] == record['source_sha256'] and demo['frames'] == 32
            assert np.load(directory/'demo_features.npy', mmap_mode='r').shape == (32, 196, 768)
            record['demo_feature_file'] = 'demo_features.npy'
            record['demo_raw_frame_indices'] = demo['raw_frame_indices']
        low = np.min([p['low'] for p in parts], axis=0)
        high = np.max([p['high'] for p in parts], axis=0)
        constant = high-low < 1e-8
        evaluation = []
        for task in tasks:
            if task['task'] not in test_tasks:
                continue
            relative = f"init_files/{task['view']}/{task['task']}.pruned_init"
            init_path = args.official_env_root/relative
            states = torch.load(init_path, map_location='cpu', weights_only=False)
            assert len(states) == 50
            for index, state in enumerate(states):
                state = np.asarray(state, dtype=np.float64)
                path = Path('evaluation')/task['task']/f'init_{index:02d}.npy'
                (args.output/path).parent.mkdir(parents=True, exist_ok=True)
                np.save(args.output/path, state)
                evaluation.append(dict(task=task['task'], view=task['view'], init_index=index,
                    initial_state=str(path), initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                    init_file=relative, init_file_sha256=digest(init_path), source=task['file'],
                    bddl=f"bddl_files/{task['view']}/{task['task']}.bddl"))
        history_capacity = max(math.ceil(550/args.history_stride),
                               max(r['observations'] for r in records if r['split'] == 'train'))
        manifest = dict(schema='fpp_libero_gen_video_full_v2', suite='combination',
            source_repo=source['source_repo'], source_revision=source['source_revision'],
            source_code_revision=args.source_revision, tasks=tasks, episodes=records, evaluation=evaluation,
            feature_dim=768, grid_size=14, max_steps=history_capacity, history_stride=args.history_stride,
            horizon=16, camera='agentview_rgb', image_transform='flipud', action_offset=1,
            demo_frames=32, demo_sampling='Uniform over original full RGB sequence, independent of stride-12 query sampling; includes final frame',
            action_alignment='source obs[j] predicts source actions[j+1:]',
            dino_model=source['dino_model'], dino_checkpoint_sha256=source['dino_checkpoint_sha256'],
            normalization=dict(center=((low+high)/2).tolist(),
                scale=np.where(constant, 1, (high-low)/2).tolist(), constant=constant.tolist()),
            training_usage='All 50 demos for every one of the 164 official training tasks; no held-out task training.',
            evaluation_protocol='Official 50 pruned initial states/task; seed 7 reset; no source-episode fixture pose override.',
            cache_identity=identity)
        temporary = args.output/'manifest.partial'
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(args.output/'manifest.json')
        print(json.dumps(dict(complete=True, train_episodes=8200, test_support_episodes=500,
                              official_test_initial_states=len(evaluation))), flush=True)
        return
    if digest(args.dino_checkpoint) != identity['dino_checkpoint_sha256']:
        raise ValueError('DINO provenance differs')
    device = torch.device(args.device)
    model = timm.create_model(source['dino_model'], pretrained=False,
        checkpoint_path=str(args.dino_checkpoint), dynamic_img_size=True).eval().to(device)
    records, lows, highs = [], [], []
    began = time.monotonic()
    for index, task in enumerate(tasks):
        if index % args.shards != args.shard:
            continue
        if (index // args.shards) % args.subshards != args.subshard:
            continue
        path = args.raw_root/task['file']
        # Recheck the complete raw file before accepting cached/new episodes.
        if digest(path) != task['source_sha256']:
            raise ValueError(f'Source checksum differs: {path}')
        split = 'train' if task['task'] in train_tasks else 'test_support'
        with h5py.File(path, 'r') as handle:
            names = sorted(handle['data'], key=lambda x: int(x.split('_')[-1]))
            if len(names) != 50:
                raise ValueError(f'Expected all 50 demos: {path}')
            for name in names:
                group = handle['data'][name]
                relative = Path('episodes')/f'task_{index:03d}'/name
                directory = args.output/relative
                marker = directory/'record.json'
                if marker.exists():
                    record = json.loads(marker.read_text())
                    if record['cache_identity'] != identity or record['split'] != split:
                        raise ValueError('Cache identity differs')
                else:
                    length = len(group['actions'])-1
                    indices = np.arange(0, length, args.history_stride)
                    if len(indices) > args.max_steps:
                        raise ValueError(f'History too short: {path}:{name}: {len(indices)}')
                    images = np.ascontiguousarray(group['obs/agentview_rgb'][indices.tolist()][:, ::-1])
                    directory.mkdir(parents=True, exist_ok=True)
                    np.save(directory/'features.npy', _features(model, images, device, args.feature_batch))
                    np.save(directory/'indices.npy', indices)
                    if split == 'train':
                        actions = np.asarray(group['actions'][1:], dtype=np.float32)
                        assert actions.shape == (length, 7) and np.isfinite(actions).all()
                        np.save(directory/'actions.npy', actions)
                    record = dict(id=f"{task['view']}/{task['task']}:{name}", task=task['task'],
                        view=task['view'], episode=name, split=split, source=task['file'],
                        source_sha256=task['source_sha256'], cache=str(relative), length=length,
                        observations=len(indices), cache_identity=identity,
                        initial_state_sha256=hashlib.sha256(np.asarray(group['states'][0], dtype=np.float64).tobytes()).hexdigest())
                    marker.write_text(json.dumps(record, indent=2))
                records.append(record)
                if split == 'train':
                    actions = np.load(directory/'actions.npy', mmap_mode='r')
                    lows.append(actions.min(0)); highs.append(actions.max(0))
        print(json.dumps(dict(shard=args.shard, task=index, episodes=len(records),
                              elapsed_s=time.monotonic()-began)), flush=True)
    part = dict(identity=identity, episodes=records, low=np.min(lows, axis=0).tolist(),
                high=np.max(highs, axis=0).tolist())
    suffix = f'.sub_{args.subshard}' if args.subshards > 1 else ''
    (args.output/f'shard_{args.shard}{suffix}.json').write_text(json.dumps(part, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['source-manifest', 'raw-root', 'output', 'dino-checkpoint', 'official-env-root']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--shards', type=int, default=8)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--subshards', type=int, default=1)
    p.add_argument('--subshard', type=int, default=0)
    p.add_argument('--merge-shard', action='store_true')
    p.add_argument('--history-stride', type=int, default=12)
    p.add_argument('--max-steps', type=int, default=64)
    p.add_argument('--feature-batch', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--source-revision', required=True)
    p.add_argument('--merge', action='store_true')
    main(p.parse_args())
