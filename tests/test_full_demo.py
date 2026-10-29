"""Full spatial support tokens and differentiable episode-level decoder K/V."""
import copy

import pytest
import torch

from fpp_libero_gen.model import VideoPrefixFPP, VideoKVCache
from test_multimodal_direct import activate, batch, config


def test_full_demo_preserves_every_patch_and_query_history_layout():
    torch.manual_seed(301)
    old=VideoPrefixFPP(config()).eval()
    model=VideoPrefixFPP(config(demo_full_patches=True)).eval()
    # Full support patches change no learned parameter/buffer shapes for this
    # dual-view query configuration, so a parent checkpoint is identifiable.
    model.load_state_dict(old.state_dict(),strict=True)
    e=model.visual
    assert e.tokens_per_time==old.visual.tokens_per_time==6
    assert e.max_steps==old.visual.max_steps==3
    assert e.demo_tokens_per_time==2*4+1+2
    data=batch()
    prompt={k:v for k,v in data.items() if k.startswith('demo')}
    with torch.no_grad():
        embedded=e.embed_demo(data['demo'],data['demo_proprio'],data['demo_actions'])
        chunks=embedded.reshape(2,3,e.demo_tokens_per_time,16)
        expected=e.input_projection(e.input_norm(data['demo']))
        expected=expected+e.spatial[None,None,None]+e.sensorimotor.view[None,None,:,None]
        expected=expected+e.segment[0]
        torch.testing.assert_close(chunks[:,:,:8],expected.flatten(2,3),rtol=0,atol=0)
        changed={k:v.clone() for k,v in prompt.items()}
        changed['demo'][0,0,1,2]+=torch.randn(12)
        actual=e.embed_demo(changed['demo'],changed['demo_proprio'],changed['demo_actions']).reshape_as(chunks)
        delta=(actual-chunks).abs().sum(-1)
        assert (delta>0).nonzero().tolist()==[[0,0,6]]
        mask=e.demo_token_mask(data['demo'],data['demo_valid'])
        assert mask.shape==(2,33) and mask.sum(1).tolist()==[11,22]


@pytest.mark.parametrize('full_patches',[False,True])
@pytest.mark.parametrize('checkpoint_direct',[False,True])
def test_shared_kv_keeps_loss_all_gradients_and_ragged_support_ownership(full_patches,checkpoint_direct):
    torch.manual_seed(302)
    settings=dict(demo_full_patches=full_patches,checkpoint_direct=checkpoint_direct,
                  checkpoint_visual=True)
    old=VideoPrefixFPP(config(**settings)).train()
    activate(old.action)
    model=VideoPrefixFPP(config(**settings,shared_demo_kv=True)).train()
    model.load_state_dict(old.state_dict())
    a={k:v.detach().clone().requires_grad_(v.is_floating_point()) for k,v in batch().items()}
    b={k:v.detach().clone().requires_grad_(v.is_floating_point()) for k,v in a.items()}
    targets=torch.randn(2,3,4,10)
    valid=torch.tensor([[1,0,1],[0,1,1]],dtype=torch.bool)
    counts=[0,0]
    def count(index):
        def hook(*_):counts[index]+=1
        return hook
    hooks=[old.action.demo_projection.register_forward_hook(count(0)),
           model.action.demo_projection.register_forward_hook(count(1))]
    try:
        torch.manual_seed(303)
        expected=old(**a,actions=targets,valid=valid,action_microbatch=3)
        torch.manual_seed(303)
        actual=model(**b,actions=targets,valid=valid,action_microbatch=3)
        # Projection count matters: otherwise repeated per-target work could
        # remain despite a numerically correct cache interface.
        assert counts==[2,1]
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        expected.backward();actual.backward()
    finally:
        for hook in hooks:hook.remove()
    reference=dict(old.named_parameters())
    for name,param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
        torch.testing.assert_close(param.grad,reference[name].grad,rtol=5e-5,atol=5e-7)
    for key in a:
        if not a[key].is_floating_point():continue
        torch.testing.assert_close(b[key].grad,a[key].grad,rtol=5e-5,atol=5e-7)
    for key in ['demo','demo_actions','demo_proprio']:
        assert b[key].grad[~b['demo_valid']].abs().max()==0
        assert b[key].grad[b['demo_valid']].abs().sum()>0
    assert model.action.demo_attention[0].key_value.weight.grad.abs().sum()>0
    assert model.visual.demo_pool_queries.grad.abs().sum()>0


def test_full_prefix_cache_padding_and_future_query_causality():
    torch.manual_seed(304)
    model=VideoPrefixFPP(config(demo_full_patches=True,shared_demo_kv=True)).eval()
    activate(model.action)
    data=batch();prompt={k:v for k,v in data.items() if k.startswith('demo')}
    noise=torch.randn(2,4,10)
    with torch.no_grad():
        full,prefix=model.visual(**data,return_demo=True)
        other=copy.deepcopy(data);other['features'][:,2]+=5;other['query_proprio'][:,2]+=5
        changed,changed_prefix=model.visual(**other,return_demo=True)
        torch.testing.assert_close(prefix,changed_prefix,rtol=0,atol=0)
        torch.testing.assert_close(full[:,:2],changed[:,:2],rtol=0,atol=0)
        cache=VideoKVCache(model.visual);cache.reset(**prompt)
        torch.testing.assert_close(cache.demo_memory,prefix,rtol=2e-5,atol=2e-6)
        for step in range(3):
            memory=cache.encode(data['features'][:,step],data['query_proprio'][:,step])
            torch.testing.assert_close(memory,full[:,step],rtol=2e-5,atol=2e-6)
            actual=model.sample_action(memory,cache,steps=3,noise=noise)
            expected=model.action.sample(full[:,step],prefix,steps=3,noise=noise,demo_valid=cache.prefix_mask)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)


def test_rejects_flags_without_their_required_conditioning_paths():
    with pytest.raises(ValueError,match='direct demo'):
        VideoPrefixFPP(config(direct_demo=False,shared_demo_kv=True))
    with pytest.raises(ValueError,match='multimodal demo'):
        VideoPrefixFPP(config(demo_input='rgb',query_input='rgb',demo_full_patches=True))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual BF16/FlexAttention required')
def test_cuda_bf16_shared_full_demo_gradients():
    torch.manual_seed(305)
    settings=dict(demo_full_patches=True,visual_dim=32,decoder_dim=32,
                  checkpoint_visual=True,checkpoint_direct=True)
    a=VideoPrefixFPP(config(**settings)).cuda().train();activate(a.action)
    b=VideoPrefixFPP(config(**settings,shared_demo_kv=True)).cuda().train()
    b.load_state_dict(a.state_dict())
    data=batch('cuda');target=torch.randn(2,3,4,10,device='cuda')
    valid=torch.tensor([[1,0,1],[0,1,1]],device='cuda',dtype=torch.bool)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        torch.manual_seed(306);expected=a(**data,actions=target,valid=valid,action_microbatch=3)
        torch.manual_seed(306);actual=b(**data,actions=target,valid=valid,action_microbatch=3)
    expected.backward();actual.backward()
    torch.testing.assert_close(actual,expected,rtol=.025,atol=.003)
    reference=dict(a.named_parameters())
    for name,param in b.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
        torch.testing.assert_close(param.grad,reference[name].grad,rtol=.06,atol=.003)
