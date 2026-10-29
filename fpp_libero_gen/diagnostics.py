"""Read-only low-frequency diagnostics for the unchanged optimization objective."""
import torch

from fpp_libero_gen.data import collate


def collate_with_ids(samples, max_steps=None):
    return collate(samples, max_steps), [
        {key: sample[key] for key in ('query_id', 'support_id', 'task')} for sample in samples]


@torch.no_grad()
def norm(tensors):
    values = [torch.linalg.vector_norm(x.detach().float()) for x in tensors if x is not None]
    return float(torch.linalg.vector_norm(torch.stack(values))) if values else 0.


def module_norms(model):
    return {name: dict(parameter_norm=norm(module.parameters()),
                       preclip_gradient_norm=norm(p.grad for p in module.parameters()))
            for name, module in [('visual', model.visual), ('action', model.action)]}


@torch.no_grad()
def update_diagnostics(model, before, ema):
    parameters = list(model.parameters())
    parameter_norm = norm(before)
    update_norm = norm(p.detach() - old for p, old in zip(parameters, before))
    result = dict(parameter_norm_before=parameter_norm, update_norm=update_norm,
                  update_to_parameter_ratio=update_norm / max(parameter_norm, 1e-30))
    if ema is not None and ema.model is not None:
        result['raw_ema_distance'] = norm(p.detach() - ema.model[name]
                                          for name, p in model.named_parameters())
        result['ema_updates'] = ema.updates
    return result
