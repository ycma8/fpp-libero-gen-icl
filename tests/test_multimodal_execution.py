import copy
import numpy as np
import pytest
import torch
from fpp_libero_gen.model import VideoPrefixFPP, VideoKVCache
from fpp_libero_gen.stability import FiniteInstabilityGuard
from fpp_libero_mem.patch_causal_dit import Attention, CrossAttentionFlowHead


def config():
    return dict(feature_dim=12,grid_size=2,demo_grid_size=1,demo_frames=3,demo_chunk_size=2,
        demo_input='bpp',query_input='bpp',query_wrist_pool='learned',demo_pooling='learned',action_qk_norm=True,action_fp32_residual=True,
        max_steps=3,visual_dim=16,visual_heads=2,visual_layers=2,ffn_ratio=2,checkpoint_visual=False,
        decoder_dim=16,decoder_heads=2,decoder_layers=2,horizon=4,action_dim=10)


def test_qk_scale_bound_and_cached_action_head_equivalence():
    torch.manual_seed(14);attn=Attention(16,2,qk_norm=True)
    q=torch.randn(2,6,16)*10000;mem=torch.randn(2,9,16)*10000
    attn.capture=True
    a=attn(q,mem);b=attn(q,kv=attn.project_memory(mem))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert attn.observation['sampled_logit_abs_max']<=8**.5+1e-5
    a.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in attn.parameters())
    h=CrossAttentionFlowHead(config()).eval()
    # Exercise all gates/projections instead of comparing two zero-initialized outputs.
    for block in h.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight,std=.02)
    torch.nn.init.normal_(h.output_projection.weight,std=.02)
    x=torch.randn(2,4,10);m=torch.randn(2,9,16);tau=torch.rand(2)
    projected=h.memory_projection(m);kv=[b.cross_attention.project_memory(b.memory_norm(projected)) for b in h.blocks]
    torch.testing.assert_close(h(x,tau,m),h(x,tau,m,cached_memory=(projected,kv)),rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_history_modalities_causality_padding_and_incremental_equivalence(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(27);cfg=config()
    if device=='cuda':cfg.update(visual_dim=32,decoder_dim=32)
    m=VideoPrefixFPP(cfg).to(device).eval()
    p=dict(demo=torch.randn(2,3,2,4,12,device=device),demo_proprio=torch.randn(2,3,11,device=device),
           demo_actions=torch.randn(2,3,2,10,device=device),demo_valid=torch.tensor([[1,1,0],[1,1,1]],device=device,dtype=torch.bool))
    x=torch.randn(2,3,2,4,12,device=device);prop=torch.randn(2,3,11,device=device)
    with torch.no_grad():
        expected=m.visual(x,query_proprio=prop,**p)
        cache=VideoKVCache(m.visual);cache.reset(**p)
        actual=torch.stack([cache.encode(x[:,i],prop[:,i]) for i in range(3)],1)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
        assert expected.shape==(2,3,6,cfg['visual_dim'])
        altered=x.clone();altered[:,2]+=10
        changed=prop.clone();changed[:,2]+=10
        torch.testing.assert_close(m.visual(altered,query_proprio=changed,**p)[:,:2],expected[:,:2],rtol=0,atol=0)
        altered=x.clone();altered[:,:,1,:,0]+=10
        assert (m.visual(altered,query_proprio=prop,**p)-expected).abs().max()>1e-4
        assert (m.visual(x,query_proprio=prop+1,**p)-expected).abs().max()>1e-4
        padded={k:v.clone() for k,v in p.items()}
        for k in ['demo','demo_proprio','demo_actions']:padded[k][0,2]=1000
        torch.testing.assert_close(m.visual(x,query_proprio=prop,**padded),expected,rtol=0,atol=0)
        if device=='cuda':
            with torch.autocast('cuda',dtype=torch.bfloat16):
                full=m.visual(x,query_proprio=prop,**p);cache.reset(**p)
                actual=torch.stack([cache.encode(x[:,i],prop[:,i]) for i in range(3)],1)
                torch.testing.assert_close(actual,full,rtol=.03,atol=.03)
    m.train();opt=torch.optim.Adam(m.parameters(),lr=.005)
    for i in range(4):
        opt.zero_grad();loss=m(x,actions=torch.randn(2,3,4,10,device=device),valid=torch.ones(2,3,device=device,dtype=torch.bool),query_proprio=prop,**p)
        loss.backward();opt.step()
    for name in ['visual.demo_pool_queries','visual.sensorimotor.proprio.0.weight','visual.sensorimotor.view']:
        g=dict(m.named_parameters())[name].grad;assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0


def test_persistent_finite_instability_guard_distinguishes_warmup_and_spikes():
    g=FiniteInstabilityGuard(after=5,window=3,patience=2)
    assert not any(g.observe(s,1,100) for s in range(5))
    for s in range(5,15):assert not g.observe(s,.01,.2)
    assert not g.observe(15,.3,100)
    assert not g.observe(16,.01,.2)
    for s in range(17,20):assert not g.observe(s,.01,.2)
    assert any(g.observe(s,.3,100) for s in range(20,30))


def test_execution_sidecar_pairing_and10d_action_alignment(tmp_path):
    import json,hashlib
    from fpp_libero_gen.data import PairedEpisodes,collate
    from fpp_libero_gen.execution import ExecutionCache
    base=tmp_path/'base';demo=tmp_path/'demo';query=tmp_path/'query'
    for p in [base,demo,query]:p.mkdir()
    episodes=[]
    for i in range(2):
        ep=dict(id=f'task:demo_{i}',task='task',split='train',cache=f'episode{i}')
        episodes.append(ep)
        for root in [base,demo,query]:(root/ep['cache']).mkdir()
        np.save(base/ep['cache']/'features.npy',np.full((2,4,12),i,np.float16))
        np.save(base/ep['cache']/'indices.npy',[0,1]);np.save(base/ep['cache']/'actions.npy',np.ones((3,7),np.float32))
        np.save(demo/ep['cache']/'demo_features.npy',np.full((2,2,4,12),i,np.float16))
        np.savez(demo/ep['cache']/'sensorimotor.npz',actions=np.ones((2,2,10),np.float32),proprio=np.ones((2,11),np.float32))
        np.save(query/ep['cache']/'wrist_features.npy',np.full((2,4,12),i+10,np.float16))
        # Already shifted by +1 during raw preparation; dataset must not shift twice.
        np.savez(query/ep['cache']/'lowdim.npz',actions=np.arange(30,dtype=np.float32).reshape(3,10),proprio=np.ones((2,11),np.float32)*i,indices=[0,1])
    m=dict(episodes=episodes,max_steps=3,horizon=2,normalization=dict(center=[0]*7,scale=[1]*7))
    raw=json.dumps(m).encode();(base/'manifest.json').write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
    norm=lambda n:dict(center=[0]*n,scale=[1]*n)
    dm=dict(schema='fpp_bpp_demo_v1',base_manifest_sha256=sha,episodes=[dict(e,chunks=2) for e in episodes],max_chunks=3,
            normalization=dict(action=norm(10),proprio=norm(11)))
    dr=json.dumps(dm).encode();(demo/'manifest.json').write_bytes(dr)
    qm=dict(schema='fpp_execution_bpp_v1',base_manifest_sha256=sha,demo_manifest_sha256=hashlib.sha256(dr).hexdigest(),episodes=episodes,
            normalization=dict(action=norm(10),proprio=norm(11)))
    (query/'manifest.json').write_text(json.dumps(qm))
    data=PairedEpisodes(base,target_steps=2,demo_cache_root=demo,execution_cache_root=query)
    s=data[0];assert s['support_id']!=s['query_id']
    assert s['features'].shape==(2,2,4,12) and s['query_proprio'].shape==(2,11)
    np.testing.assert_array_equal(s['actions'][0,0].numpy(),np.arange(10))
    np.testing.assert_array_equal(s['actions'][1,0].numpy(),np.arange(10,20))
    b=collate([s,data[1]],max_steps=3)
    assert b['features'].shape==(2,3,2,4,12) and b['actions'].shape==(2,3,2,10)
    assert not b['query_proprio'][:,2].any()
    with pytest.raises(ValueError,match='timestamps'):data.execution_cache.load(episodes[0]['id'],np.array([0,2]))
    with pytest.raises(ValueError,match='identity'):ExecutionCache(query,'bad')


def test_guard_history_survives_checkpoint_resume():
    a=FiniteInstabilityGuard(after=0,window=3,patience=2)
    for step in range(10):a.observe(step,.01,.1)
    a.observe(10,.2,100);a.observe(11,.2,100)
    b=FiniteInstabilityGuard(after=0,window=3,patience=2);b.load_state_dict(a.state_dict())
    assert a.observe(12,.2,100)==b.observe(12,.2,100)==True


def test_action_checkpointing_preserves_nonzero_loss_and_gradients():
    torch.manual_seed(98);m=VideoPrefixFPP(config()).train()
    for p in m.action.parameters():torch.nn.init.normal_(p,std=.02)
    c=copy.deepcopy(m);c.visual.config=dict(c.visual.config,checkpoint_action=True)
    batch=dict(features=torch.randn(1,3,2,4,12),query_proprio=torch.randn(1,3,11),demo=torch.randn(1,3,2,4,12),
               demo_actions=torch.randn(1,3,2,10),demo_proprio=torch.randn(1,3,11),demo_valid=torch.ones(1,3,dtype=torch.bool),
               actions=torch.randn(1,3,4,10),valid=torch.ones(1,3,dtype=torch.bool))
    torch.manual_seed(23);a=m(**batch);a.backward()
    torch.manual_seed(23);b=c(**batch);b.backward()
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for (name,p),(_,q) in zip(m.named_parameters(),c.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-7,msg=name)


def test_official_proprio_and_native_action_transform_parity():
    mod=pytest.importorskip('behavior_prompting.train_network.model.common.rotation_transformer')
    from fpp_libero_gen.execution import policy_proprio,native_action
    from fpp_libero_gen.sensorimotor import rotation_6d
    rng=np.random.default_rng(18);aa=rng.normal(size=(10,3)).astype(np.float32)*.2
    quaternion=mod.RotationTransformer(from_rep='axis_angle',to_rep='quaternion').forward(aa)
    reference=mod.RotationTransformer(from_rep='quaternion',to_rep='rotation_6d').forward(quaternion)
    for i in range(10):
        obs=dict(robot0_eef_quat=quaternion[i,[1,2,3,0]][None],robot0_eef_pos=np.ones((1,3),np.float32),robot0_gripper_qpos=np.zeros((1,2),np.float32))
        np.testing.assert_allclose(policy_proprio(obs)[3:9],reference[i],rtol=0,atol=1e-6)
    actions=np.concatenate((np.ones((10,3)),rotation_6d(aa),np.ones((10,1))),-1).astype(np.float32)
    native=native_action(actions);np.testing.assert_allclose(native[:,3:6],aa,atol=1e-6)
    assert native.shape==(10,7)
