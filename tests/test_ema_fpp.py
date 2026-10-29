import copy

import torch

from fpp_libero_gen.diagnostics import module_norms, update_diagnostics
from fpp_libero_gen.ema import ModelEMA
from fpp_libero_gen.model import VideoPrefixFPP


def test_ema_and_full_diagnostics_preserve_fpp_training_updates():
    config = dict(feature_dim=12, grid_size=2, demo_grid_size=1, demo_frames=3,
                  max_steps=3, visual_dim=16, visual_heads=2, visual_layers=2,
                  ffn_ratio=2, checkpoint_visual=False, decoder_dim=16,
                  decoder_heads=2, decoder_layers=2, horizon=4, action_dim=7)
    def train(enabled):
        torch.manual_seed(42)
        model = VideoPrefixFPP(config).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .95))
        ema = ModelEMA(.9, 2)
        for step in range(1, 5):
            features, demo = torch.randn(2, 3, 4, 12), torch.randn(2, 3, 4, 12)
            actions = torch.randn(2, 3, 4, 7)
            valid = torch.tensor([[True, False, True], [False, True, False]])
            optimizer.zero_grad()
            loss = model(features=features, demo=demo, actions=actions, valid=valid)
            loss.backward()
            if enabled: module_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            before = [p.detach().clone() for p in model.parameters()] if enabled else None
            optimizer.step()
            if enabled:
                ema.update(model, step)
                diagnostic = update_diagnostics(model, before, ema)
                assert diagnostic['update_norm'] > 0
        return copy.deepcopy(model.state_dict()), torch.get_rng_state()
    baseline, baseline_rng = train(False)
    actual, actual_rng = train(True)
    for name in baseline:
        torch.testing.assert_close(baseline[name], actual[name], rtol=0, atol=0)
    assert torch.equal(baseline_rng, actual_rng)
