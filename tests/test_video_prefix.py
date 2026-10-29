import numpy as np
import json
import pytest
import torch

from fpp_libero_gen.model import VideoPrefixFPP, VideoKVCache, demo_causal_mask
from fpp_libero_gen.data import PairedEpisodes, collate
from fpp_libero_gen.prompt_controls import transform_demo


def config():
    return dict(feature_dim=12, grid_size=2, demo_grid_size=1, demo_frames=3,
        max_steps=4, visual_dim=16, visual_heads=2, visual_layers=2, ffn_ratio=2,
        checkpoint_visual=False, decoder_dim=16, decoder_heads=2, decoder_layers=2,
        horizon=4, action_dim=7)


def test_prefix_and_query_attention_boundaries():
    q = torch.arange(15)
    mask = demo_causal_mask(q[:, None], q[None], 3, 4)
    assert mask[:3, :3].all() and not mask[:3, 3:].any()
    assert mask[:, :3].all()
    assert mask[3:7, 3:7].all() and not mask[3:7, 7:].any()


def test_causality_demo_condition_and_incremental_equivalence():
    torch.manual_seed(31)
    model = VideoPrefixFPP(config()).eval()
    x, demo = torch.randn(2, 4, 4, 12), torch.randn(2, 3, 4, 12)
    with torch.no_grad():
        expected = model.visual(x, demo)
        changed = x.clone()
        changed[:, 2:] += 10*torch.randn_like(changed[:, 2:])
        actual = model.visual(changed, demo)
        torch.testing.assert_close(expected[:, :2], actual[:, :2])
        assert not torch.allclose(expected, model.visual(x, demo+torch.randn_like(demo)))
        cache = VideoKVCache(model.visual)
        cache.reset(demo)
        for t in range(4):
            torch.testing.assert_close(cache.encode(x[:, t]), expected[:, t], rtol=2e-5, atol=2e-6)
        with pytest.raises(ValueError):
            cache.encode(x[:, 0])


def test_gradients_reach_demo_from_query_action_loss():
    torch.manual_seed(12)
    model = VideoPrefixFPP(config())
    torch.nn.init.normal_(model.action.output_projection.weight, std=.02)
    for block in model.action.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=.02)
    demo = torch.randn(1, 3, 4, 12, requires_grad=True)
    loss = model(torch.randn(1, 3, 4, 12), demo, torch.randn(1, 3, 4, 7),
                 torch.tensor([[True, False, True]]))
    loss.backward()
    assert torch.isfinite(loss) and demo.grad.abs().sum() > 0


def test_pairing_excludes_query_episode_and_test_task(tmp_path):
    episodes=[]
    for task, split in [('train_task','train'),('novel_task','test_support')]:
        for i in range(2):
            cache=tmp_path/f'{task}_{i}'
            cache.mkdir()
            np.save(cache/'features.npy',np.full((3,4,12),i,dtype=np.float16))
            np.save(cache/'actions.npy',np.arange(7*6,dtype=np.float32).reshape(6,7))
            np.save(cache/'indices.npy',np.array([0,2,4]))
            episodes.append(dict(task=task,split=split,id=f'{task}_{i}',cache=cache.name))
    manifest=dict(episodes=episodes,history_stride=2,max_steps=4,horizon=4,
        normalization=dict(center=[0]*7,scale=[1]*7,constant=[False]*7))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    dataset=PairedEpisodes(tmp_path,demo_frames=3,target_steps=2)
    assert len(dataset)==2 and set(dataset.by_task)=={'train_task'}
    row=dataset[0]
    assert row['query_id']=='train_task_0' and row['support_id']=='train_task_1'
    assert row['demo'].eq(1).all() and row['features'].eq(0).all()
    assert row['valid'].sum()==2
    np.testing.assert_array_equal(row['actions'][1,0],np.arange(14,21))
    batch=collate([row],max_steps=4)
    assert not batch['valid'][0,-1]
    # Full-data runs sample support frames directly from the full source video,
    # independently of the query observation stride. Support actions are unused.
    for i, ep in enumerate(episodes):
        ep['demo_feature_file'] = 'demo_features.npy'
        np.save(tmp_path/ep['cache']/'demo_features.npy', np.full((3,4,12),i+10,dtype=np.float16))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'train_task_1'/'actions.npy').unlink()
    new = PairedEpisodes(tmp_path,demo_frames=3,target_steps=2)[0]
    assert new['features'].eq(0).all() and new['demo'].eq(11).all()


@pytest.mark.parametrize('pooling', ['average', 'learned'])
def test_direct_demo_cache_preserves_prefix_and_causality(pooling):
    torch.manual_seed(61)
    cfg = dict(config(), direct_demo=True, demo_pooling=pooling)
    model = VideoPrefixFPP(cfg).eval()
    features, demo = torch.randn(2, 4, 4, 12), torch.randn(2, 3, 4, 12)
    with torch.no_grad():
        query, prefix = model.visual(features, demo, return_demo=True)
        altered = features.clone()
        altered[:, 2:] += 10 * torch.randn_like(altered[:, 2:])
        changed_query, changed_prefix = model.visual(altered, demo, return_demo=True)
        torch.testing.assert_close(prefix, changed_prefix)
        torch.testing.assert_close(query[:, :2], changed_query[:, :2])
        cache = VideoKVCache(model.visual)
        cache.reset(demo)
        torch.testing.assert_close(prefix, cache.demo_memory, rtol=2e-5, atol=2e-6)
        for i in range(4):
            torch.testing.assert_close(cache.encode(features[:, i]), query[:, i], rtol=2e-5, atol=2e-6)


def activate_head(head):
    torch.nn.init.normal_(head.output_projection.weight, std=.1)
    for block in head.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=.1)
    for gate in head.demo_gate:
        torch.nn.init.normal_(gate[-1].weight, std=.1)


def test_direct_path_changes_actions_with_fixed_current_memory_and_cached_sampling_matches():
    torch.manual_seed(63)
    model = VideoPrefixFPP(dict(config(), direct_demo=True)).eval()
    activate_head(model.action)
    memory, prefix = torch.randn(2, 4, 16), torch.randn(2, 3, 16)
    noise, tau = torch.randn(2, 4, 7), torch.tensor([.2, .6])
    with torch.no_grad():
        action = model.action(noise, tau, memory, prefix)
        altered = model.action(noise, tau, memory, prefix + torch.randn_like(prefix))
        assert not torch.allclose(action, altered)
        expected = noise.clone()
        for step in range(3):
            expected += model.action(expected, torch.full((2,), step/3), memory, prefix)/3
        actual = model.action.sample(memory, prefix, steps=3, noise=noise)
        torch.testing.assert_close(expected, actual)


def test_learned_pool_and_direct_attention_receive_finite_gradients():
    torch.manual_seed(64)
    model = VideoPrefixFPP(dict(config(), direct_demo=True, demo_pooling='learned'))
    activate_head(model.action)
    demo = torch.randn(2, 3, 4, 12, requires_grad=True)
    loss = model(torch.randn(2, 3, 4, 12), demo, torch.randn(2, 3, 4, 7),
                 torch.tensor([[True, False, True], [False, True, True]]), action_microbatch=2)
    loss.backward()
    for value in [demo, model.visual.demo_pool_queries,
                  model.visual.demo_pool_attention.key_value.weight,
                  model.action.demo_attention[0].key_value.weight]:
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_prompt_controls_preserve_length_and_do_not_advance_policy_rng():
    demo = np.arange(32*4*12).reshape(32, 4, 12)
    np.random.seed(65)
    expected = np.random.random(4)
    np.random.seed(65)
    shuffled, order = transform_demo(demo, 'shuffled', seed=66)
    np.testing.assert_array_equal(np.random.random(4), expected)
    assert sorted(order) == list(range(32)) and order != list(range(32))
    np.testing.assert_array_equal(shuffled, demo[order])
    np.testing.assert_array_equal(shuffled, transform_demo(demo, 'shuffled', seed=66)[0])
    last, order = transform_demo(demo, 'last_frame', seed=66)
    assert order == [31]*32 and last.shape == demo.shape
    np.testing.assert_array_equal(last, np.repeat(demo[-1:], 32, axis=0))
    np.testing.assert_array_equal(transform_demo(demo, 'correct', 66)[0], demo)
    assert not transform_demo(demo, 'blank', 66)[0].any()
