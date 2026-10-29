import copy
import io

import pytest
import torch
from torch import nn

from fpp_libero_gen.ema import ModelEMA, select_weights, weight_sha256


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1., 2.]))
        self.register_buffer('counter', torch.tensor(0))
        self.register_buffer('scale', torch.tensor(1.))


def test_initialization_formula_and_buffer_semantics():
    model = Tiny()
    ema = ModelEMA(decay=.5, start_step=2)
    ema.update(model, 1)
    assert ema.model is None
    with torch.no_grad(): model.weight.fill_(4)
    ema.update(model, 2)
    assert ema.updates == 0 and ema.initialized_step == 2
    with torch.no_grad():
        model.weight.fill_(8)
        model.counter.fill_(7)
        model.scale.fill_(9)
    ema.update(model, 3)
    torch.testing.assert_close(ema.model['weight'], torch.tensor([6., 6.]))
    assert ema.model['weight'].dtype == torch.float32
    assert ema.model['counter'] == 7 and ema.model['scale'] == 9
    assert ema.updates == 1 and ema.last_step == 3
    with pytest.raises(ValueError, match='exactly once'): ema.update(model, 3)


def test_ema_and_diagnostics_do_not_change_raw_updates_or_rng():
    def train(enabled):
        torch.manual_seed(42)
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        ema = ModelEMA(.9, 2)
        for step in range(1, 9):
            x, target = torch.randn(5, 3), torch.randn(5, 2)
            optimizer.zero_grad()
            (model(x)-target).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if enabled:
                before_rng = torch.get_rng_state().clone()
                ema.update(model, step)
                assert torch.equal(before_rng, torch.get_rng_state())
        return model.state_dict(), optimizer.state_dict(), torch.get_rng_state()
    baseline, a, rng_a = train(False)
    actual, b, rng_b = train(True)
    for k in baseline: torch.testing.assert_close(baseline[k], actual[k], rtol=0, atol=0)
    for key in a['state']:
        for field in a['state'][key]:
            torch.testing.assert_close(a['state'][key][field], b['state'][key][field], rtol=0, atol=0)
    assert torch.equal(rng_a, rng_b)


def test_checkpoint_roundtrip_and_explicit_selection():
    model = Tiny(); ema = ModelEMA(.9, 2)
    for step in range(1, 5):
        with torch.no_grad(): model.weight.add_(1)
        ema.update(model, step)
    saved = dict(step=4, model=copy.deepcopy(model.state_dict()), ema=ema.state_dict())
    stream = io.BytesIO(); torch.save(saved, stream); stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored = ModelEMA(.9, 2); restored.load_state_dict(saved['ema'], model)
    with torch.no_grad(): model.weight.add_(1)
    restored.update(model, 5); ema.update(model, 5)
    for name in ema.model: torch.testing.assert_close(restored.model[name], ema.model[name], rtol=0, atol=0)
    assert weight_sha256(select_weights(saved, 'raw')) != weight_sha256(select_weights(saved, 'ema'))
    assert select_weights(dict(model=saved['model'])) is saved['model']
    with pytest.raises(ValueError, match='absent'): select_weights(dict(model=saved['model']), 'ema')
    uninitialized = dict(step=1, model=model.state_dict(), ema=ModelEMA(.9, 2).state_dict())
    with pytest.raises(ValueError, match='initialized'): select_weights(uninitialized, 'ema')
    with pytest.raises(ValueError, match='config'): ModelEMA(.8, 2).load_state_dict(saved['ema'], model)
    saved['step'] = 5
    with pytest.raises(ValueError, match='optimizer step'): select_weights(saved, 'ema')
