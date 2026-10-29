import hashlib
import json

import numpy as np
import pytest
import torch

from fpp_libero_gen.model import VideoPrefixFPP, VideoKVCache
from fpp_libero_gen.sensorimotor import rotation_6d, convert_fields, chunk_fields, range_spec, SensorimotorCache, transform_prompt
from fpp_libero_gen.data import PairedEpisodes, collate


def config():
    return dict(feature_dim=12,grid_size=2,demo_grid_size=1,demo_frames=3,demo_chunk_size=2,
        demo_input='bpp',max_steps=3,visual_dim=16,visual_heads=2,visual_layers=2,ffn_ratio=2,
        checkpoint_visual=False,decoder_dim=16,decoder_heads=2,decoder_layers=2,horizon=4,action_dim=7)


def prompt():
    return dict(demo=torch.randn(2,3,2,4,12),demo_actions=torch.randn(2,3,2,10),
        demo_proprio=torch.randn(2,3,11),demo_valid=torch.tensor([[True,True,False],[True,True,True]]))


@pytest.mark.parametrize('length',[1,19,20,21,40,357])
def test_rotation_chunking_keeps_full_rate_actions_and_last_partial_chunk(length):
    np.testing.assert_allclose(rotation_6d(np.array([[0,0,0],[0,0,np.pi/2]])),
        [[1,0,0,0,1,0],[0,-1,0,1,0,0]],atol=1e-7)
    raw=np.arange(length*7,dtype=np.float32).reshape(length,7)/100
    act,prop=convert_fields(raw,np.ones((length,3)),np.zeros((length,3)),np.ones((length,2)))
    indices,chunked,proprio=chunk_fields(act,prop)
    assert len(indices)==(length+19)//20
    np.testing.assert_array_equal(indices,np.arange(0,length,20))
    np.testing.assert_array_equal(chunked.reshape(-1,10)[:length],act)
    assert not chunked.reshape(-1,10)[length:].any()
    np.testing.assert_array_equal(proprio,prop[indices])


def test_incremental_equivalence_padding_invariance_and_query_causality():
    torch.manual_seed(71)
    model=VideoPrefixFPP(config()).eval()
    p=prompt();x=torch.randn(2,3,4,12)
    with torch.no_grad():
        expected=model.visual(x,**p)
        cache=VideoKVCache(model.visual);cache.reset(**p)
        actual=torch.stack([cache.encode(x[:,i]) for i in range(3)],1)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        changed={k:v.clone() for k,v in p.items()}
        for key in ['demo','demo_actions','demo_proprio']:
            changed[key][0,2]=100
        torch.testing.assert_close(model.visual(x,**changed),expected,rtol=0,atol=0)
        future=x.clone();future[:,2]+=5
        torch.testing.assert_close(model.visual(future,**p)[:,:2],expected[:,:2],rtol=0,atol=0)
        for key in ['demo','demo_actions','demo_proprio']:
            changed={k:v.clone() for k,v in p.items()};changed[key][:,0,...,0]+=3
            assert (model.visual(x,**changed)-expected).abs().max()>1e-5


def test_multimodal_gradient_and_padded_chunk_mask_validation():
    model=VideoPrefixFPP(config()).train();p=prompt()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    # The original flow head intentionally starts with a zero output and zero gates.
    for _ in range(4):
        optimizer.zero_grad()
        loss=model(features=torch.randn(2,3,4,12),actions=torch.randn(2,3,4,7),
            valid=torch.ones((2,3),dtype=torch.bool),**p)
        loss.backward()
        optimizer.step()
    for name,param in model.visual.sensorimotor.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
        assert param.grad.abs().sum()>0,name
    p['demo_valid'][0]=False
    with pytest.raises(ValueError,match='real chunks'):
        model.visual(torch.randn(2,3,4,12),**p)


def test_task_safe_pairing_sidecar_identity_and_synchronized_controls(tmp_path):
    base=tmp_path/'base';side=tmp_path/'side';base.mkdir();side.mkdir()
    episodes=[];records=[]
    norm=dict(center=[0]*7,scale=[1]*7,constant=[False]*7)
    for i in range(2):
        cache=f'episode{i}';(base/cache).mkdir();(side/cache).mkdir()
        np.save(base/cache/'features.npy',np.ones((2,4,12),np.float16))
        np.save(base/cache/'indices.npy',np.array([0,1]))
        np.save(base/cache/'actions.npy',np.ones((3,7),np.float32)*i)
        # No RGB-only demo file is provided: the sidecar path must be used.
        ep=dict(id=f'train/t:{i}',task='t',split='train',cache=cache);episodes.append(ep)
        np.save(side/cache/'demo_features.npy',np.full((2,2,4,12),i,np.float16))
        np.savez(side/cache/'sensorimotor.npz',actions=np.full((2,2,10),i,np.float32),proprio=np.full((2,11),i,np.float32))
        records.append(dict(ep,chunks=2))
    manifest=dict(episodes=episodes,max_steps=3,horizon=4,normalization=norm)
    raw=json.dumps(manifest).encode();(base/'manifest.json').write_bytes(raw)
    side_manifest=dict(schema='fpp_bpp_demo_v1',base_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        episodes=records,max_chunks=3,normalization=dict(action=range_spec([-1]*10,[1]*10,[0,1,2]),
                                                        proprio=range_spec([-1]*11,[1]*11,[0,1,2,9,10])))
    (side/'manifest.json').write_text(json.dumps(side_manifest))
    data=PairedEpisodes(base,target_steps=2,demo_cache_root=side)
    sample=data[0]
    assert sample['query_id']!=sample['support_id'] and sample['demo_actions'][0,0,0]==1
    batch=collate([sample,data[1]],max_steps=3)
    assert batch['demo_valid'].shape==(2,3) and batch['demo'].shape==(2,3,2,4,12)
    with pytest.raises(ValueError,match='identities'):
        SensorimotorCache(side,'wrong')
    p=SensorimotorCache(side).load(episodes[0]['id'])
    for key in ['demo','demo_actions','demo_proprio']:
        p[key][1]=2
    p2,order=transform_prompt(p,'shuffled',3)
    for key in ['demo','demo_actions','demo_proprio']:
        np.testing.assert_array_equal(p2[key][:2],p[key][order])
    blank,_=transform_prompt(p,'blank',3)
    assert not any(blank[k].any() for k in ['demo','demo_actions','demo_proprio'])
    np.testing.assert_array_equal(blank['demo_valid'],p['demo_valid'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires actual CUDA FlexAttention')
def test_cuda_padding_mask_and_incremental_cache_match_cpu():
    torch.manual_seed(83)
    cfg=config();cfg.update(visual_dim=32,decoder_dim=32)
    model=VideoPrefixFPP(cfg).eval();p=prompt();x=torch.randn(2,3,4,12)
    with torch.no_grad():
        reference=model.visual(x,**p)
        model=model.cuda();p={k:v.cuda() for k,v in p.items()};x=x.cuda()
        full=model.visual(x,**p)
        torch.testing.assert_close(full.cpu(),reference,atol=2e-5,rtol=2e-5)
        cache=VideoKVCache(model.visual);cache.reset(**p)
        actual=torch.stack([cache.encode(x[:,i]) for i in range(3)],1)
        torch.testing.assert_close(actual,full,atol=2e-5,rtol=2e-5)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prefix=model.visual.embed_demo(p['demo'],p['demo_proprio'],p['demo_actions'])
            assert prefix.dtype==model.visual.embed_query(x).dtype==torch.bfloat16
            full=model.visual(x,**p)
            cache.reset(**p)
            actual=torch.stack([cache.encode(x[:,i]) for i in range(3)],1)
            torch.testing.assert_close(actual,full,atol=.02,rtol=.02)
        # Same shape, different valid lengths: a stale per-batch mask must not be reused.
        p['demo_valid'][0,2]=True;p['demo_valid'][1,2]=False
        full=model.visual(x,**p)
        cache.reset(**p)
        actual=torch.stack([cache.encode(x[:,i]) for i in range(3)],1)
        torch.testing.assert_close(actual,full,atol=2e-5,rtol=2e-5)
