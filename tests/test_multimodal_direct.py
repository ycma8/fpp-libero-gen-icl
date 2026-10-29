"""Multimodal direct support access without padded keys or future-query leakage."""
import copy

import pytest
import torch

from fpp_libero_gen.model import VideoKVCache, VideoPrefixFPP


def config(**extra):
    cfg = dict(feature_dim=12, grid_size=2, demo_grid_size=1, demo_frames=3,
                demo_chunk_size=2, demo_input='bpp', query_input='bpp',
                demo_pooling='learned', query_wrist_pool='learned',
                max_steps=3, visual_dim=16, visual_heads=2, visual_layers=2,
                ffn_ratio=2, checkpoint_visual=False, decoder_dim=16,
                decoder_heads=2, decoder_layers=2, horizon=4, action_dim=10,
                direct_demo=True, action_qk_norm=True, action_fp32_residual=True)
    cfg.update(extra)
    return cfg


def batch(device='cpu'):
    return dict(features=torch.randn(2, 3, 2, 4, 12, device=device),
                query_proprio=torch.randn(2, 3, 11, device=device),
                demo=torch.randn(2, 3, 2, 4, 12, device=device),
                demo_actions=torch.randn(2, 3, 2, 10, device=device),
                demo_proprio=torch.randn(2, 3, 11, device=device),
                demo_valid=torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.bool, device=device))


def activate(head):
    torch.nn.init.normal_(head.output_projection.weight, std=.1)
    for block in head.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=.1)
    for gate in head.demo_gate:
        torch.nn.init.normal_(gate[-1].weight, std=.1)


def test_direct_head_masks_padding_and_reads_each_demo_modality_with_fixed_query():
    torch.manual_seed(201)
    model = VideoPrefixFPP(config()).eval()
    activate(model.action)
    data = batch()
    prompt = {k: v for k, v in data.items() if k.startswith('demo')}
    memory = torch.randn(2, model.visual.tokens_per_time, 16)
    noisy, tau = torch.randn(2, 4, 10), torch.tensor([.3, .7])
    mask = model.visual.demo_token_mask(data['demo'], data['demo_valid'])
    with torch.no_grad():
        _, prefix = model.visual(**data, return_demo=True)
        expected = model.action(noisy, tau, memory, prefix, demo_valid=mask)
        corrupt = prefix.clone()
        corrupt[~mask] = torch.randn_like(corrupt[~mask]) * 100
        torch.testing.assert_close(model.action(noisy, tau, memory, corrupt, demo_valid=mask),
                                   expected, rtol=0, atol=0)
        assert not torch.allclose(model.action(noisy, tau, memory, corrupt), expected)
        # Hold current query memory fixed so changes must reach the direct path.
        for key in ['demo', 'demo_actions', 'demo_proprio']:
            changed = {k: v.clone() for k, v in prompt.items()}
            changed[key][:, 0] += torch.randn_like(changed[key][:, 0]) * 3
            _, new_prefix = model.visual(data['features'], return_demo=True,
                                        query_proprio=data['query_proprio'], **changed)
            actual = model.action(noisy, tau, memory, new_prefix, demo_valid=mask)
            assert (actual - expected).abs().max() > 1e-5, key
    with pytest.raises(ValueError, match='boolean token-valid'):
        model.action(noisy, tau, memory, prefix, demo_valid=mask.float())
    with pytest.raises(ValueError, match='at least one valid'):
        model.action(noisy, tau, memory, prefix, demo_valid=torch.zeros_like(mask))


@pytest.mark.parametrize('checkpoint_direct', [False, True])
def test_training_owners_padding_and_gradients(checkpoint_direct):
    torch.manual_seed(202)
    model = VideoPrefixFPP(config(checkpoint_direct=checkpoint_direct)).train()
    activate(model.action)
    other = copy.deepcopy(model)
    data = batch()
    changed = {k: v.clone() for k, v in data.items()}
    for key in ['demo', 'demo_actions', 'demo_proprio']:
        changed[key][~data['demo_valid']] = 50
    target = torch.randn(2, 3, 4, 10)
    valid = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)
    torch.manual_seed(203)
    expected = model(**data, actions=target, valid=valid, action_microbatch=3)
    expected.backward()
    torch.manual_seed(203)
    actual = other(**changed, actions=target, valid=valid, action_microbatch=3)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    params = dict(model.named_parameters())
    for name, param in other.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        torch.testing.assert_close(param.grad, params[name].grad, rtol=0, atol=0)
    for name in ['action.demo_attention.0.key_value.weight', 'visual.demo_pool_queries',
                 'visual.sensorimotor.action.0.weight', 'visual.sensorimotor.proprio.0.weight']:
        assert params[name].grad.abs().sum() > 0, name
    for module in [model.action.demo_attention[0].key_value,
                   model.visual.sensorimotor.action, model.visual.sensorimotor.proprio]:
        assert any(p.grad.abs().sum() > 0 for p in module.parameters())


def test_incremental_prefix_mask_causality_and_cached_flow_match_full_forward():
    torch.manual_seed(204)
    model = VideoPrefixFPP(config()).eval()
    activate(model.action)
    data = batch()
    prompt = {k: v for k, v in data.items() if k.startswith('demo')}
    noise = torch.randn(2, 4, 10)
    with torch.no_grad():
        full, prefix = model.visual(**data, return_demo=True)
        future = {k: v.clone() for k, v in data.items()}
        future['features'][:, 2] += 5
        future['query_proprio'][:, 2] += 5
        altered, altered_prefix = model.visual(**future, return_demo=True)
        torch.testing.assert_close(full[:, :2], altered[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(prefix, altered_prefix, rtol=0, atol=0)
        cache = VideoKVCache(model.visual)
        cache.reset(**prompt)
        torch.testing.assert_close(cache.demo_memory, prefix, rtol=2e-5, atol=2e-6)
        for step in range(3):
            memory = cache.encode(data['features'][:, step], data['query_proprio'][:, step])
            torch.testing.assert_close(memory, full[:, step], rtol=2e-5, atol=2e-6)
            expected = noise.clone()
            for flow_step in range(3):
                tau = torch.full((2,), flow_step / 3)
                expected += model.action(expected, tau, full[:, step], prefix,
                                         demo_valid=cache.prefix_mask).float() / 3
            actual = model.sample_action(memory, cache, steps=3, noise=noise)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires actual CUDA BF16/FlexAttention')
def test_cuda_bf16_direct_mask_cache_and_bounded_attention():
    torch.manual_seed(205)
    # Torch 2.7 CUDA FlexAttention requires at least16 channels per head.
    model = VideoPrefixFPP(config(visual_dim=32, decoder_dim=32)).cuda().eval()
    activate(model.action)
    data = batch('cuda')
    prompt = {k: v for k, v in data.items() if k.startswith('demo')}
    noise = torch.randn(2, 4, 10, device='cuda')
    for attention in model.action.demo_attention:
        attention.capture = True
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        full, prefix = model.visual(**data, return_demo=True)
        cache = VideoKVCache(model.visual)
        cache.reset(**prompt)
        for step in range(3):
            memory = cache.encode(data['features'][:, step], data['query_proprio'][:, step])
            torch.testing.assert_close(memory, full[:, step], atol=.025, rtol=.025)
            actual = model.sample_action(memory, cache, steps=3, noise=noise)
            expected = model.action.sample(full[:, step], prefix, steps=3, noise=noise,
                                           demo_valid=cache.prefix_mask)
            torch.testing.assert_close(actual, expected, atol=.025, rtol=.025)
            assert torch.isfinite(actual).all()
        for attention in model.action.demo_attention:
            assert attention.qk_norm
            assert attention.observation['sampled_logit_abs_max'] <= 16 ** .5 + .03
        dtypes = []
        hook = model.action.blocks[0].register_forward_pre_hook(lambda _, args: dtypes.append(args[0].dtype))
        try:
            model.action(noise, torch.tensor([.2, .8], device='cuda'), full[:, 0], prefix,
                         demo_valid=cache.prefix_mask)
        finally:
            hook.remove()
        assert dtypes == [torch.float32]
