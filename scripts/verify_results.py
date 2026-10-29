"""Verify the standalone release using Python's standard library only."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_official_video import verify, require

def main():
    base=ROOT/'results/reference'
    provenance=json.loads((base/'provenance.json').read_text())
    raw=gzip.decompress((base/'canonical-manifest.json.gz').read_bytes())
    require(hashlib.sha256(raw).hexdigest()==provenance['canonical_manifest_sha256'],'Canonical manifest differs')
    canonical=json.loads(raw)
    trials={(r['task'],r['init_index']):r for r in canonical['evaluation']}
    require(len(trials)==500,'Canonical500 differs')
    tasks=sorted({r['task'] for r in canonical['evaluation']})
    support={(r['task'],r['episode']):r['id'] for r in canonical['episodes'] if r['split']=='test_support'}
    outputs={};identities={}
    for condition in ['correct','wrong']:
        folder=base/condition;score=json.loads((folder/'performance.json').read_text())
        result=verify(folder,folder/'performance.json')
        rows=[json.loads(s) for p in sorted(folder.glob('trials_shard*.jsonl')) for s in p.read_text().splitlines() if s.strip()]
        require({(r['task'],r['init_index']) for r in rows}==set(trials),'Trial membership differs')
        for row in rows:
            key=(row['task'],row['init_index']);target=trials[key];i=tasks.index(row['task'])
            require(row['initial_state_sha256']==target['initial_state_sha256'],'Canonical state differs')
            require(row['seed']==4242+1000*i+row['init_index'],'Inference seed differs')
            prompt_task=tasks[(i+1)%len(tasks)] if condition=='wrong' else row['task']
            require(row['support_id']==support[prompt_task,f"demo_{row['init_index']%10}"],'Support differs')
        for p in folder.glob('*shard*'):
            entry=provenance['export']['source_files'][str(p.relative_to(ROOT))]
            if not entry['metadata_checkpoint_path_redacted']:
                require(hashlib.sha256(p.read_bytes()).hexdigest()==entry['original_sha256'],'Original trial/summary bytes differ')
        require(score['checkpoint_sha256']==provenance['checkpoint_sha256'],'Checkpoint differs')
        require(score['protocol']['weight_sha256']==provenance['ema_weight_sha256'],'EMA weights differ')
        require(result['successes']==provenance['reported_conditions'][condition]['successes'],'Published count differs')
        outputs[condition]=dict(successes=result['successes'],trials=500,success_rate=result['success_rate'])
        identities[condition]={(r['task'],r['initial_state_sha256'],r['seed']) for r in rows}
    require(identities['correct']==identities['wrong'],'Unmatched control')
    for name,digest in provenance['core_source_sha256'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Reported core implementation differs: '+name)
    result=dict(verified=True,conditions=outputs,total_trials=1000,matched_control=True,core_source_bytes_verified=True)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':main()
