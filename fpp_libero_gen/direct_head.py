"""An independent demo cross-attention path in every flow decoder block."""
import torch
from torch import nn

from fpp_libero_mem.patch_causal_dit import Attention, CrossAttentionFlowHead, sinusoidal


class DirectDemoFlowHead(CrossAttentionFlowHead):
    def __init__(self, config):
        super().__init__(config)
        dim = config['decoder_dim']
        self.demo_projection = nn.Linear(config['visual_dim'], dim)
        self.demo_attention = nn.ModuleList([
            Attention(dim, config['decoder_heads'], qk_norm=config.get('action_qk_norm', False))
            for _ in self.blocks])
        self.demo_query_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in self.blocks])
        self.demo_memory_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in self.blocks])
        self.demo_gate = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(dim, dim)) for _ in self.blocks])
        for gate in self.demo_gate:
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    @staticmethod
    def _demo_mask(demo_memory, demo_valid):
        if demo_valid is None:
            return None
        if demo_valid.dtype != torch.bool or demo_valid.shape != demo_memory.shape[:2]:
            raise ValueError('Direct demo needs a boolean token-valid mask matching its memory')
        if not demo_valid.any(1).all():
            raise ValueError('Direct demo needs at least one valid token per example')
        return demo_valid[:, None, None, :]

    def project_demo_kv(self, demo_memory):
        """One differentiable projection per support episode and decoder layer."""
        demo = self.demo_projection(demo_memory)
        return tuple(tensor for attention, norm in zip(self.demo_attention, self.demo_memory_norm)
                     for tensor in attention.project_memory(norm(demo)))

    def forward(self, noisy, tau, memory, demo_memory, cached_memory=None, demo_valid=None,
                demo_kv=None, demo_owners=None):
        time = self.time_mlp(sinusoidal(tau*1000, self.flow_dim))
        x = self.input_projection(noisy)
        x = x + self.action_position.to(x.dtype)[None]
        if self.fp32_residual:
            x = x.float()
        mask = self._demo_mask(demo_memory, demo_valid)
        if demo_kv is not None:
            if cached_memory is not None or len(demo_kv) != 2*len(self.blocks):
                raise ValueError('Shared demo K/V must match decoder layers and cannot mix with inference cache')
        if demo_owners is not None:
            if demo_kv is None or demo_owners.dtype != torch.long or demo_owners.shape != (len(noisy),):
                raise ValueError('Shared demo owners need one integer support index per target')
            if ((demo_owners < 0) | (demo_owners >= len(demo_memory))).any():
                raise ValueError('Shared demo owner index is outside its episode batch')
            mask = None if mask is None else mask.index_select(0, demo_owners)
        if cached_memory is None:
            projected = self.memory_projection(memory)
            demo = self.demo_projection(demo_memory) if demo_kv is None else None
        else:
            projected, demo, _, _ = cached_memory
        for i, block in enumerate(self.blocks):
            x = block(x, time, projected, None if cached_memory is None else cached_memory[2][i])
            if demo_kv is not None:
                kv = demo_kv[2*i:2*i+2]
                if demo_owners is not None:
                    kv = tuple(tensor.index_select(0, demo_owners) for tensor in kv)
            else:
                kv = None if cached_memory is None else cached_memory[3][i]
            update = self.demo_attention[i](self.demo_query_norm[i](x),
                self.demo_memory_norm[i](demo) if kv is None else None, kv, attn_mask=mask)
            x = x + self.demo_gate[i](time)[:, None] * update
        shift, scale = self.output_modulation(time).chunk(2, dim=-1)
        return self.output_projection(self.output_norm(x)*(1+scale[:, None]) + shift[:, None])

    @torch.no_grad()
    def sample(self, memory, demo_memory, steps=10, noise=None, demo_valid=None):
        if steps < 1:
            raise ValueError('Flow integration requires positive steps')
        self._demo_mask(demo_memory, demo_valid)
        projected, demo = self.memory_projection(memory), self.demo_projection(demo_memory)
        current_kv = [b.cross_attention.project_memory(b.memory_norm(projected)) for b in self.blocks]
        demo_kv = [a.project_memory(n(demo)) for a, n in zip(self.demo_attention, self.demo_memory_norm)]
        cached = projected, demo, current_kv, demo_kv
        x = torch.randn(len(memory), self.horizon, self.action_dim, device=memory.device) if noise is None else noise.clone()
        for step in range(steps):
            tau = torch.full((len(x),), step/steps, device=x.device)
            x = x + self(x, tau, memory, demo_memory, cached_memory=cached,
                         demo_valid=demo_valid).float()/steps
        return x
