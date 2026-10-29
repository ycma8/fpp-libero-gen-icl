"""Prepare BPP demo modalities in a new sidecar; original query cache is read-only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from fpp_libero_gen.sensorimotor import convert_fields, chunk_fields, range_spec
from prepare_video import digest


def main(a):
    a.output.mkdir(parents=True, exist_ok=True)
    raw = (a.data_root/'manifest.json').read_bytes()
    base = json.loads(raw)
    identity = dict(base_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        dino_checkpoint_sha256=base['dino_checkpoint_sha256'], chunk_size=20,
        cameras=['agentview_rgb','eye_in_hand_rgb'], source_revision=a.source_revision,
        shards=a.shards)
    if a.merge:
        parts = [json.loads((a.output/f'shard_{i}.json').read_text()) for i in range(a.shards)]
        if not all(p['identity']==identity for p in parts):
            raise ValueError('Cache shard identities differ')
        records = sorted([e for p in parts for e in p['episodes']], key=lambda e:e['id'])
        if {e['id'] for e in records} != {e['id'] for e in base['episodes']} or len(records)!=len(base['episodes']):
            raise ValueError('Missing or duplicated demos')
        train = [e for e in records if e['split']=='train']
        if len(train)!=8200 or len({e['task'] for e in train})!=164:
            raise ValueError('Full training split required')
        stats = {}
        for key, selected in [('action',[0,1,2]),('proprio',[0,1,2,9,10])]:
            low = np.min([e['statistics'][key]['low'] for e in train],axis=0)
            high = np.max([e['statistics'][key]['high'] for e in train],axis=0)
            stats[key] = range_spec(low,high,selected)
        for e in records:
            p = a.output/e['cache']
            if digest(p/'sensorimotor.npz') != e['lowdim_sha256'] or digest(p/'demo_features.npy') != e['features_sha256']:
                raise ValueError('Cache contents changed')
        result = dict(schema='fpp_bpp_demo_v1', **identity, episodes=records,
            normalization=stats, normalization_split='train only: all8200 source training demos',
            max_chunks=max(e['chunks'] for e in records),
            image_transform='flipud; FPP DINOv3 pad/resize/normalization',
            proprio_fields=['ee_pos3','ee_ori_rot6d6','gripper_states2'],
            action_fields=['delta_position3','delta_orientation_rot6d6','gripper_command1'],
            demo_alignment='BPP source convention: obs[j],proprio[j],raw actions[j:j+20]; no query offset applied',
            padding='Partial final20-action chunk zero padded AFTER conversion to10D; whole padded chunks attention masked',
            query_inputs='Unchanged original single-view RGB history and offset+1 action targets',
            bpp_config_sha256=digest(a.bpp_config),
            bpp_source_revision='96294c35705a18823ff2c850d37fbb1e12c21d85')
        tmp=a.output/'manifest.partial';tmp.write_text(json.dumps(result,indent=2));tmp.replace(a.output/'manifest.json')
        print(json.dumps(dict(complete=True,episodes=len(records),max_chunks=result['max_chunks'],identity=identity)),flush=True)
        return
    import timm
    from scripts.prepare_cache import _features
    if digest(a.dino_checkpoint)!=base['dino_checkpoint_sha256']:
        raise ValueError('DINO differs from query cache')
    device=torch.device(a.device)
    model=timm.create_model(base['dino_model'],pretrained=False,
        checkpoint_path=str(a.dino_checkpoint),dynamic_img_size=True).eval().to(device)
    records=[];began=time.monotonic()
    for task_index,task in enumerate(base['tasks']):
        if task_index%a.shards!=a.shard:continue
        source=a.raw_root/task['file']
        episodes=[e for e in base['episodes'] if e['task']==task['task']]
        verified=a.data_root/episodes[0]['cache']/'record.json'
        prior=json.loads(verified.read_text())
        stat=source.stat()
        reused=(stat.st_size==task['bytes'] and stat.st_mtime_ns<=verified.stat().st_mtime_ns
                and prior['source_sha256']==task['source_sha256'])
        if not reused and digest(source)!=task['source_sha256']:
            raise ValueError('Raw source SHA differs')
        with h5py.File(source,'r') as h:
            for e in episodes:
                directory=a.output/e['cache'];marker=directory/'record.json'
                if marker.exists():
                    record=json.loads(marker.read_text())
                    if record['identity']!=identity or record['source_sha256']!=e['source_sha256']:
                        raise ValueError('Existing cache identity differs')
                else:
                    g=h['data'][e['episode']]
                    action,proprio=convert_fields(g['actions'][:],g['obs/ee_pos'][:],
                                                 g['obs/ee_ori'][:],g['obs/gripper_states'][:])
                    indices,chunked,positions=chunk_fields(action,proprio)
                    images=np.stack([g['obs'][key][:][indices,::-1] for key in identity['cameras']],axis=1)
                    features=_features(model,np.ascontiguousarray(images.reshape(-1,*images.shape[2:])),device,a.feature_batch)
                    features=features.reshape(len(indices),2,196,768)
                    directory.mkdir(parents=True,exist_ok=True)
                    np.save(directory/'demo_features.npy',features)
                    np.savez(directory/'sensorimotor.npz',actions=chunked,proprio=positions,indices=indices)
                    statistics={key:dict(low=x.min(0).tolist(),high=x.max(0).tolist())
                                for key,x in [('action',action),('proprio',proprio)]}
                    record=dict(id=e['id'],task=e['task'],split=e['split'],cache=e['cache'],
                        source=e['source'],source_sha256=e['source_sha256'],source_length=len(action),
                        raw_frame_indices=indices.tolist(),chunks=len(indices),identity=identity,
                        source_verification='prior SHA and unchanged size/mtime' if reused else 'fresh SHA256',
                        statistics=statistics,
                        lowdim_sha256=digest(directory/'sensorimotor.npz'),
                        features_sha256=digest(directory/'demo_features.npy'))
                    tmp=directory/'record.partial';tmp.write_text(json.dumps(record,indent=2));tmp.replace(marker)
                records.append(record)
        print(json.dumps(dict(shard=a.shard,task=task_index,episodes=len(records),elapsed_s=time.monotonic()-began)),flush=True)
    tmp=a.output/f'shard_{a.shard}.partial'
    tmp.write_text(json.dumps(dict(identity=identity,episodes=records),indent=2));tmp.replace(a.output/f'shard_{a.shard}.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data-root','raw-root','output','dino-checkpoint','bpp-config']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-revision',required=True)
    p.add_argument('--shards',type=int,default=8)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--feature-batch',type=int,default=64)
    p.add_argument('--merge',action='store_true')
    main(p.parse_args())
