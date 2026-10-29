import argparse
import json
from pathlib import Path
import train
from train_video import build_config
ROOT=Path(__file__).resolve().parents[1]

def test_method_defaults_match_reported_model():
    values=train.arguments([])
    config=json.loads((ROOT/'configs/reference.json').read_text())
    a=argparse.Namespace(width=512,visual_layers=6,decoder_layers=8,demo_grid=7,demo_frames=32,
        direct_demo=True,shared_demo_kv=True,demo_full_patches=True,demo_pooling='learned',
        demo_input='bpp',query_input='bpp',action_qk_norm=True,action_fp32_residual=True,
        query_wrist_pool='learned',checkpoint_action=False)
    model=build_config(dict(feature_dim=768,grid_size=14,max_steps=46,horizon=16),a,dict(max_chunks=18,chunk_size=20))
    assert model==config['model']
    assert '--direct-demo' in values and '--shared-demo-kv' in values and '--demo-full-patches' in values
    assert values[values.index('--steps')+1]=='10000'
    assert values[values.index('--schedule-steps')+1]=='50000'
    assert values[values.index('--warmup')+1]=='2000'
    assert values[values.index('--ema-decay')+1]=='0.999'
    assert '--resume' not in values

def test_explicit_flags_are_not_duplicated():
    result=train.arguments(['--steps=20','--batch','8'])
    assert '--steps' not in result and result.count('--batch')==1

def test_reference_records():
    from scripts.verify_results import main
    assert main()['verified']
