"""Cache 32 true source-video frames independently of the query control stride."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import timm
import torch

from prepare_video import digest
from scripts.prepare_cache import _features


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['source-manifest', 'raw-root', 'output', 'dino-checkpoint']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--shards', type=int, default=8)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    source = json.loads(a.source_manifest.read_text())
    checkpoint_sha = digest(a.dino_checkpoint)
    assert checkpoint_sha == source['dino_checkpoint_sha256']
    device = torch.device(a.device)
    model = timm.create_model(source['dino_model'], pretrained=False,
        checkpoint_path=str(a.dino_checkpoint), dynamic_img_size=True).eval().to(device)
    count, began = 0, time.monotonic()
    for index, task in enumerate(source['tasks']):
        if index % a.shards != a.shard:
            continue
        path = a.raw_root/task['file']
        # The query-cache pass already hashed every complete source. Reuse that
        # provenance only while the raw source's size and mtime remain unchanged
        # since its verified cache record. Otherwise re-hash the raw file.
        verified_record = a.output/'episodes'/f'task_{index:03d}'/'demo_0'/'record.json'
        reuse_verification = False
        if verified_record.exists():
            prior = json.loads(verified_record.read_text())
            raw_stat = path.stat()
            reuse_verification = (prior['source_sha256'] == task['source_sha256'] and
                raw_stat.st_size == task['bytes'] and raw_stat.st_mtime_ns <= verified_record.stat().st_mtime_ns)
        if not reuse_verification:
            assert digest(path) == task['source_sha256']
        with h5py.File(path, 'r') as handle:
            names = sorted(handle['data'], key=lambda x: int(x.split('_')[-1]))
            assert len(names) == 50
            for name in names:
                directory = a.output/'episodes'/f'task_{index:03d}'/name
                marker = directory/'demo_record.json'
                if marker.exists():
                    record = json.loads(marker.read_text())
                    assert record['checkpoint_sha256'] == checkpoint_sha
                    assert record['source_sha256'] == task['source_sha256']
                else:
                    source_rgb = handle['data'][name]['obs/agentview_rgb']
                    indices = np.linspace(0, len(source_rgb)-1, 32).round().astype(int)
                    unique, inverse = np.unique(indices, return_inverse=True)
                    # These HDF5 videos are contiguous, not chunked. Sequential
                    # episode reads avoid many expensive random reads; selected
                    # pixels are identical to HDF5 fancy indexing.
                    images = np.ascontiguousarray(source_rgb[:][unique][:, ::-1])
                    features = _features(model, images, device, 64)[inverse]
                    directory.mkdir(parents=True, exist_ok=True)
                    np.save(directory/'demo_features.npy', features)
                    record = dict(frames=32, raw_frame_indices=indices.tolist(),
                        checkpoint_sha256=checkpoint_sha, source_sha256=task['source_sha256'],
                        source_verification=('verified query-cache SHA; raw size and mtime unchanged' if reuse_verification else 'fresh SHA-256'),
                        image_transform='flipud', camera='agentview_rgb',
                        note='No support actions or proprioception read; source final RGB frame retained.')
                    temporary = directory/'demo_record.partial'
                    temporary.write_text(json.dumps(record, indent=2))
                    temporary.replace(marker)
                count += 1
        print(json.dumps(dict(shard=a.shard, task=index, episodes=count,
                              elapsed_s=time.monotonic()-began)), flush=True)
    temporary = a.output/f'prefix_shard{a.shard}.partial'
    temporary.write_text(json.dumps(dict(shard=a.shard, episodes=count, complete=True)))
    temporary.replace(a.output/f'prefix_shard{a.shard}.json')
