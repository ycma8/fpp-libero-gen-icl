"""BPP execution modalities beside the unchanged full-history RGB cache.

Stage only sparse wrist RGB and low-dimensional fields on the raw-data host;
extract wrist DINO features on the training host. Neither stage modifies source
caches. All normalization comes from the audited train-only BPP demo cache.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    t=p.with_suffix('.partial');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)


def stage_task(job):
    task,episodes,base_root,raw_root,out,identity=job
    from fpp_libero_gen.sensorimotor import convert_fields
    base_root,raw_root,out=map(Path,[base_root,raw_root,out])
    source=raw_root/task['file'];prior=base_root/episodes[0]['cache']/'record.json'
    stat=source.stat();old=json.loads(prior.read_text())
    reused=(stat.st_size==task['bytes'] and stat.st_mtime_ns<=prior.stat().st_mtime_ns and old['source_sha256']==task['source_sha256'])
    if not reused and digest(source)!=task['source_sha256']:raise ValueError('Raw source hash changed')
    records=[]
    with h5py.File(source,'r') as h:
        for e in episodes:
            directory=out/e['cache'];directory.mkdir(parents=True,exist_ok=True);marker=directory/'stage.json'
            if marker.exists():
                record=json.loads(marker.read_text())
                if record['identity']!=identity:raise ValueError('Existing execution stage identity differs')
            else:
                g=h['data'][e['episode']]
                actions,proprio=convert_fields(g['actions'][:],g['obs/ee_pos'][:],g['obs/ee_ori'][:],g['obs/gripper_states'][:])
                indices=np.load(base_root/e['cache']/'indices.npy')
                assert len(actions)-1==e['length'] and len(indices)==e['observations']
                wrist=np.ascontiguousarray(g['obs/eye_in_hand_rgb'][indices.tolist()][:,::-1])
                np.save(directory/'wrist_rgb.npy',wrist)
                np.savez(directory/'lowdim.npz',actions=actions[1:],proprio=proprio[indices],indices=indices,
                         native_actions=np.asarray(g['actions'][1:],dtype=np.float32))
                record=dict(e,identity=identity,wrist_rgb_sha256=digest(directory/'wrist_rgb.npy'),
                            lowdim_sha256=digest(directory/'lowdim.npz'),raw_verification='prior SHA plus unchanged size/mtime' if reused else 'fresh full SHA')
                write(marker,record)
            records.append(record)
    return records


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    base_raw=(a.data_root/'manifest.json').read_bytes();base=json.loads(base_raw)
    demo_raw=(a.demo_cache_root/'manifest.json').read_bytes();demo=json.loads(demo_raw)
    identity=dict(base_manifest_sha256=hashlib.sha256(base_raw).hexdigest(),demo_manifest_sha256=hashlib.sha256(demo_raw).hexdigest(),
                  source_revision=a.source_revision,dino_checkpoint_sha256=base['dino_checkpoint_sha256'],history_stride=12,action_offset=1)
    assert demo['base_manifest_sha256']==identity['base_manifest_sha256'] and base['history_stride']==12
    if a.stage_raw:
        jobs=[(t,[e for e in base['episodes'] if e['task']==t['task']],str(a.data_root),str(a.raw_root),str(a.output),identity) for t in base['tasks']]
        records=[];began=time.monotonic()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for part in pool.map(stage_task,jobs):
                records+=part;print(json.dumps(dict(stage_episodes=len(records),elapsed_s=time.monotonic()-began)),flush=True)
        assert len(records)==8700 and sum(e['split']=='train' for e in records)==8200
        norm=dict(demo['normalization']['action'],constant=[False]*10)
        write(a.output/'stage_manifest.json',dict(schema='fpp_execution_stage_v1',identity=identity,episodes=records,
            normalization=dict(action=norm,proprio=demo['normalization']['proprio']),
            alignment='Query obs/proprio at raw row j; predict raw action[j+1:] in10D; demo retains original BPP same-row chunks'))
        return
    stage=json.loads((a.output/'stage_manifest.json').read_text())
    assert stage['identity']==identity
    if a.merge:
        parts=[json.loads((a.output/f'features_shard{i}.json').read_text()) for i in range(a.shards)]
        assert all(p['identity']==identity for p in parts)
        records=[e for p in parts for e in p['episodes']]
        assert len(records)==8700 and {e['id'] for e in records}=={e['id'] for e in base['episodes']}
        write(a.output/'manifest.json',dict(schema='fpp_execution_bpp_v1',**identity,episodes=records,
              normalization=stage['normalization'],camera_keys=['agentview_rgb','eye_in_hand_rgb'],action_dim=10,proprio_dim=11,
              feature_dim=768,grid_size=14,normalization_split='All8200 training demos only; audited BPP train-only statistics',alignment=stage['alignment']))
        return
    import torch,timm
    from scripts.prepare_cache import _features
    assert digest(a.dino_checkpoint)==identity['dino_checkpoint_sha256']
    device=torch.device(a.device);model=timm.create_model(base['dino_model'],pretrained=False,checkpoint_path=str(a.dino_checkpoint),dynamic_img_size=True).eval().to(device)
    tasks={t['task'] for i,t in enumerate(base['tasks']) if i%a.shards==a.shard};records=[];began=time.monotonic()
    for e in stage['episodes']:
        if e['task'] not in tasks:continue
        directory=a.output/e['cache'];p=directory/'features.json'
        if p.exists():
            record=json.loads(p.read_text());assert record['identity']==identity
        else:
            assert digest(directory/'wrist_rgb.npy')==e['wrist_rgb_sha256'] and digest(directory/'lowdim.npz')==e['lowdim_sha256']
            images=np.load(directory/'wrist_rgb.npy')
            features=_features(model,images,device,a.feature_batch)
            np.save(directory/'wrist_features.npy',features)
            record=dict(e,wrist_features_sha256=digest(directory/'wrist_features.npy'))
            write(p,record)
        records.append(record)
        if len(records)%50==0:print(json.dumps(dict(shard=a.shard,episodes=len(records),elapsed_s=time.monotonic()-began)),flush=True)
    write(a.output/f'features_shard{a.shard}.json',dict(identity=identity,episodes=records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['data-root','demo-cache-root','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--raw-root',type=Path);p.add_argument('--stage-raw',action='store_true');p.add_argument('--merge',action='store_true')
    p.add_argument('--dino-checkpoint',type=Path);p.add_argument('--source-revision',required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--shards',type=int,default=8);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--device',default='cuda');p.add_argument('--feature-batch',type=int,default=64)
    main(p.parse_args())
