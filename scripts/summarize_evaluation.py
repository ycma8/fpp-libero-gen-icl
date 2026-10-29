"""Aggregate all completed official shards before independent record checks."""
import argparse
import hashlib
import json
from pathlib import Path

def summarize(root):
    protocols=sorted(root.glob('protocol_shard*.json'))
    if len(protocols)!=8:raise ValueError('Require all eight logical shards')
    protocol=json.loads(protocols[0].read_text())
    if any(json.loads(p.read_text())!=protocol for p in protocols):raise ValueError('Mixed protocols')
    rows=[]
    for shard in range(8):
        p=root/f'trials_shard{shard}.jsonl';summary=json.loads((root/f'summary_shard{shard}.json').read_text())
        part=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        if len(part)!=summary['trials'] or sum(r['success'] for r in part)!=summary['successes']:raise ValueError('Incomplete shard')
        rows.extend(part)
    if len(rows)!=len({(r['task'],r['init_index']) for r in rows}) or len(rows)!=500:raise ValueError('Require500 distinct trials')
    tasks=sorted({r['task'] for r in rows})
    if len(tasks)!=10:raise ValueError('Require10 tasks')
    per_task={}
    for task in tasks:
        part=[r for r in rows if r['task']==task]
        if len(part)!=50:raise ValueError('Require50/task')
        n=sum(r['success'] for r in part);per_task[task]=dict(trials=50,successes=n,success_rate=n/50)
    checkpoint=Path(protocol['checkpoint'])
    if not checkpoint.is_file():raise FileNotFoundError('Use the recorded checkpoint path for its SHA256')
    h=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    successes=sum(r['success'] for r in rows)
    return dict(checkpoint_step=protocol['checkpoint_step'],checkpoint_sha256=h.hexdigest(),condition=protocol['condition'],weight_source=protocol['weight_source'],trials=500,successes=successes,success_rate=successes/500,per_task=per_task,protocol=protocol)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(summarize(a.evaluation),indent=2)+'\n')
