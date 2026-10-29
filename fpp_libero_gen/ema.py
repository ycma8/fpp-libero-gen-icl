"""Optional FP32 parameter EMA; raw parameters and RNG streams stay untouched."""
import hashlib
import json

import torch


class ModelEMA:
    def __init__(self, decay=0.999, start_step=2000):
        if not 0 < decay < 1 or start_step < 0:
            raise ValueError('EMA requires 0 < decay < 1 and nonnegative start_step')
        self.decay, self.start_step = float(decay), int(start_step)
        self.model = None
        self.initialized_step = None
        self.last_step = 0
        self.updates = 0

    @torch.no_grad()
    def update(self, model, step):
        if step != self.last_step + 1:
            raise ValueError('EMA must follow each optimizer update exactly once')
        self.last_step = step
        if step < self.start_step:
            return
        state = model.state_dict()
        parameters = dict(model.named_parameters())
        if self.model is None:
            self.model = {name: value.detach().clone().float() if name in parameters and value.is_floating_point()
                          else value.detach().clone() for name, value in state.items()}
            self.initialized_step = step
            return
        for name, value in state.items():
            if name in parameters and value.is_floating_point():
                self.model[name].lerp_(value.detach().float(), 1 - self.decay)
            else:
                self.model[name].copy_(value)
        self.updates += 1

    def state_dict(self):
        return dict(format='fpp_parameter_ema_v1', decay=self.decay, start_step=self.start_step,
                    initialized_step=self.initialized_step, last_step=self.last_step,
                    updates=self.updates, model=self.model)

    def load_state_dict(self, state, model):
        if state['format'] != 'fpp_parameter_ema_v1' or state['decay'] != self.decay or state['start_step'] != self.start_step:
            raise ValueError('EMA format/config changed')
        self.last_step, self.updates = state['last_step'], state['updates']
        self.initialized_step = state['initialized_step']
        if state['model'] is None:
            if self.initialized_step is not None or self.updates:
                raise ValueError('Uninitialized EMA has update history')
            self.model = None
            return
        current = model.state_dict()
        if set(current) != set(state['model']):
            raise ValueError('EMA parameter/buffer keys changed')
        self.model = {}
        for name, value in current.items():
            shadow = state['model'][name]
            if shadow.shape != value.shape:
                raise ValueError('EMA tensor shape changed')
            self.model[name] = shadow.detach().to(device=value.device).clone()


def select_weights(checkpoint, choice='raw'):
    """Return explicit inference state; absent EMA never silently falls back."""
    if choice == 'raw':
        return checkpoint['model']
    if choice != 'ema':
        raise ValueError('Weights must be raw or ema')
    state = checkpoint.get('ema')
    if state is None or state.get('model') is None:
        raise ValueError('Requested EMA weights are absent or not initialized')
    if state['last_step'] != checkpoint['step']:
        raise ValueError('EMA does not match checkpoint optimizer step')
    return state['model']


def weight_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
