"""Incremental policy inference in native simulator action coordinates."""
import numpy as np
import torch
from .data import denormalize
from .incremental import VisualKVCache
from .prefix_model import PrefixPatchDiT


class Policy:
    def __init__(self, checkpoint, normalization=None, device='cuda'):
        # Load trusted, locally produced checkpoints only.
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.config = saved['config']
        self.stats = normalization if normalization is not None else saved['normalization']
        self.device = torch.device(device)
        self.model = PrefixPatchDiT(self.config).to(self.device).eval()
        self.model.load_state_dict(saved['model'])
        self.cache = VisualKVCache(self.model)
        self.metadata = {k: saved.get(k) for k in ['history_stride', 'text_features_sha256',
                         'dino_checkpoint_sha256', 'data_manifest_sha256']}

    def reset(self):
        self.cache.reset()

    @torch.inference_mode()
    def predict(self, features, text, flow_steps=10):
        if flow_steps < 1:
            raise ValueError('flow_steps must be positive')
        features = torch.as_tensor(np.asarray(features).copy(), device=self.device, dtype=torch.float32)[None]
        text = torch.as_tensor(np.asarray(text).copy(), device=self.device, dtype=torch.float32)[None]
        memory = self.cache.encode(features, text)
        actions = self.model.action.sample(memory, steps=flow_steps)[0].float().cpu().numpy()
        actions = denormalize(actions, self.stats)
        if not np.isfinite(actions).all():
            raise FloatingPointError('Policy produced non-finite actions')
        return np.clip(actions, -1, 1).astype(np.float32)
