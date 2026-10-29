"""Full-patch block-causal visual Transformer and frame-local cross-attention FM.

Frozen DINO extraction is external. No state, text, task-trained vision weights,
CLS pooling, or action-target communication between observation times is used.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sinusoidal(position: Tensor, dim: int) -> Tensor:
    """Encode positions without sharing observation, action, and flow-time indices."""
    frequency = torch.exp(torch.arange(dim // 2, device=position.device).float()
                          * (-math.log(10000.0) / (dim // 2)))
    phase = position.float()[..., None] * frequency
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class Attention(nn.Module):
    """Multihead attention with explicit query and memory projections."""

    def __init__(self, dim: int, heads: int, qk_norm: bool = False) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError('Attention width must be divisible by head count')
        self.heads = heads
        self.qk_norm = qk_norm
        self.capture = False
        self.observation = {}
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.output = nn.Linear(dim, dim)

    def project_memory(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, dim = memory.shape
        kv = self.key_value(memory).view(batch, length, 2, self.heads, dim // self.heads)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        return self.normalize(k), v

    def normalize(self, value: Tensor) -> Tensor:
        if not self.qk_norm:
            return value
        return (F.normalize(value.float(), dim=-1, eps=1e-6) * math.sqrt(value.shape[-1])).to(value.dtype)

    def forward(self, query: Tensor, memory: Tensor | None = None,
                kv: tuple[Tensor, Tensor] | None = None,
                attn_mask: Tensor | None = None) -> Tensor:
        batch, length, dim = query.shape
        q = self.query(query).view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        raw_q_rms = q.detach().float().square().mean().sqrt() if self.capture else None
        q = self.normalize(q)
        if kv is None:
            kv = self.project_memory(query if memory is None else memory)
        if self.capture:
            with torch.no_grad():
                logits = q[:, :, :16].float() @ kv[0][:, :, :64].float().transpose(-1, -2) / math.sqrt(q.shape[-1])
                logp = logits.log_softmax(-1)
                self.observation = dict(raw_q_rms=float(raw_q_rms), key_rms=float(kv[0].float().square().mean().sqrt()),
                    sampled_logit_abs_max=float(logits.abs().max()), sampled_entropy=float(-(logp.exp()*logp).sum(-1).mean()))
        output = F.scaled_dot_product_attention(q, *kv, attn_mask=attn_mask)
        return self.output(output.transpose(1, 2).reshape(batch, length, dim))


class VisualBlock(nn.Module):
    """Pre-norm attention and FFN; efficient attention is supplied by the encoder."""

    def __init__(self, dim: int, heads: int, ratio: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm_attention = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ratio), nn.GELU(approximate='tanh'),
                                 nn.Linear(dim * ratio, dim))

    def forward(self, x: Tensor, attention: Any, mask: Any) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(self.norm_attention(x)).view(
            batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        update = attention(*qkv.unbind(0), block_mask=mask)
        x = x + self.projection(update.transpose(1, 2).reshape(batch, length, dim))
        return x + self.ffn(self.norm_ffn(x))


class PatchTemporalEncoder(nn.Module):
    """Every patch attends all cameras at the same time and all preceding times."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        dim = config['visual_dim']
        self.tokens_per_time = config['cameras'] * config['grid_size'] ** 2
        self.max_steps = config['max_steps']
        self.checkpoint_layers = config['checkpoint_visual']
        self.input_norm = nn.LayerNorm(config['feature_dim'])
        self.input_projection = nn.Linear(config['feature_dim'], dim)
        self.camera = nn.Parameter(torch.randn(config['cameras'], dim) * 0.02)
        grid = config['grid_size']
        yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
        spatial = torch.cat([sinusoidal(yy.flatten(), dim // 2),
                             sinusoidal(xx.flatten(), dim // 2)], dim=-1)
        self.register_buffer('spatial_position', spatial)
        self.register_buffer('time_position', sinusoidal(torch.arange(self.max_steps), dim))
        self.blocks = nn.ModuleList([VisualBlock(dim, config['visual_heads'], config['ffn_ratio'])
                                     for _ in range(config['visual_layers'])])
        self.output_norm = nn.LayerNorm(dim)
        self._block_masks: dict = {}
        self._flex = None

    def _attention(self, steps: int, device: torch.device) -> tuple[Any, Any]:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        if self._flex is None:
            self._flex = torch.compile(flex_attention, dynamic=True)
        key = (steps, str(device))
        if key not in self._block_masks:
            tokens = self.tokens_per_time

            def staircase(batch, head, q_idx, kv_idx):
                return q_idx // tokens >= kv_idx // tokens

            self._block_masks[key] = create_block_mask(
                staircase, B=None, H=None, Q_LEN=steps*tokens, KV_LEN=steps*tokens,
                device=str(device), _compile=True)
        return self._flex, self._block_masks[key]

    def forward(self, features: Tensor) -> Tensor:
        batch, steps, tokens, _ = features.shape
        if tokens != self.tokens_per_time or steps > self.max_steps:
            raise ValueError(f'Invalid visual layout: {features.shape}')
        positions = (self.camera[:, None] + self.spatial_position[None]).flatten(0, 1)
        x = self.input_projection(self.input_norm(features))
        x = x + positions.to(x.dtype)[None, None] + self.time_position[:steps].to(x.dtype)[None, :, None]
        x = x.flatten(1, 2)
        if features.device.type == 'cuda':
            attention, mask = self._attention(steps, features.device)
        else:
            # Small CPU correctness tests only; never construct a dense long-context mask.
            if steps*tokens > 4096:
                raise ValueError('Full-patch long contexts require CUDA FlexAttention')
            times = torch.arange(steps, device=x.device).repeat_interleave(tokens)
            mask = times[:, None] >= times[None, :]

            def attention(q, k, v, block_mask):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)

        for block in self.blocks:
            if self.training and self.checkpoint_layers:
                x = checkpoint(block, x, attention, mask, use_reentrant=False)
            else:
                x = block(x, attention, mask)
        return self.output_norm(x).view(batch, steps, tokens, -1)


class ActionDiTBlock(nn.Module):
    """Bidirectional action self-attention, visual cross-attention, and AdaLN-Zero."""

    def __init__(self, dim: int, heads: int, ratio: int, qk_norm: bool = False) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3)])
        self.self_attention = Attention(dim, heads, qk_norm=qk_norm)
        self.cross_attention = Attention(dim, heads, qk_norm=qk_norm)
        self.memory_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ratio*dim), nn.GELU(approximate='tanh'),
                                 nn.Linear(ratio*dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9*dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: Tensor, time: Tensor, memory: Tensor,
                kv: tuple[Tensor, Tensor] | None = None) -> Tensor:
        parameters = self.modulation(time).chunk(9, dim=-1)
        for index in range(3):
            shift, scale, gate = [value[:, None] for value in parameters[3*index:3*index+3]]
            normalized = self.norms[index](x) * (1+scale) + shift
            if index == 0:
                update = self.self_attention(normalized)
            elif index == 1:
                update = self.cross_attention(normalized, self.memory_norm(memory), kv)
            else:
                update = self.ffn(normalized)
            x = x + gate * update
        return x


class CrossAttentionFlowHead(nn.Module):
    """Flow velocity with one token per future action, conditioned on current patches."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        dim = config['decoder_dim']
        self.horizon, self.action_dim = config['horizon'], config['action_dim']
        self.flow_dim = dim
        self.fp32_residual = config.get('action_fp32_residual', False)
        self.input_projection = nn.Linear(self.action_dim, dim)
        self.memory_projection = nn.Linear(config['visual_dim'], dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer('action_position', sinusoidal(torch.arange(self.horizon), dim))
        self.blocks = nn.ModuleList([ActionDiTBlock(dim, config['decoder_heads'], config['ffn_ratio'],
                                                  qk_norm=config.get('action_qk_norm', False))
                                     for _ in range(config['decoder_layers'])])
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2*dim))
        self.output_projection = nn.Linear(dim, self.action_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.zeros_(self.output_modulation[-1].weight)
        nn.init.zeros_(self.output_modulation[-1].bias)

    def forward(self, noisy: Tensor, tau: Tensor, memory: Tensor,
                cached_memory: tuple | None = None) -> Tensor:
        time = self.time_mlp(sinusoidal(tau*1000, self.flow_dim))
        x = self.input_projection(noisy)
        x = x + self.action_position.to(x.dtype)[None]
        if self.fp32_residual:
            x = x.float()
        memory = self.memory_projection(memory) if cached_memory is None else cached_memory[0]
        for index, block in enumerate(self.blocks):
            x = block(x, time, memory, None if cached_memory is None else cached_memory[1][index])
        shift, scale = self.output_modulation(time).chunk(2, dim=-1)
        return self.output_projection(self.output_norm(x)*(1+scale[:, None]) + shift[:, None])

    @torch.no_grad()
    def sample(self, memory: Tensor, steps: int = 10, noise: Tensor | None = None) -> Tensor:
        """Euler integration from Gaussian noise (tau=0) to action (tau=1)."""
        projected = self.memory_projection(memory)
        kv = [block.cross_attention.project_memory(block.memory_norm(projected)) for block in self.blocks]
        x = torch.randn(len(memory), self.horizon, self.action_dim, device=memory.device) if noise is None else noise.clone()
        for step in range(steps):
            tau = torch.full((len(x),), step/steps, device=x.device)
            x = x + self(x, tau, memory, cached_memory=(projected, kv)).float()/steps
        return x


class PatchCausalDiT(nn.Module):
    """Independent action FM losses at all valid observation times in full context."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = dict(config)
        self.visual = PatchTemporalEncoder(config)
        self.action = CrossAttentionFlowHead(config)

    def forward(self, features: Tensor, actions: Tensor, valid: Tensor,
                action_microbatch: int = 32) -> Tensor:
        visual = self.visual(features)[valid]
        target = actions[valid].float()
        # Each observation has an independent flow time and noise; no label is
        # passed to the temporal encoder or any other observation's action chunk.
        tau = torch.rand(len(target), device=target.device)
        noise = torch.randn_like(target)
        noisy = (1-tau[:, None, None])*noise + tau[:, None, None]*target
        losses = []
        for start in range(0, len(target), action_microbatch):
            end = start+action_microbatch
            predicted = self.action(noisy[start:end], tau[start:end], visual[start:end])
            losses.append((predicted.float() - (target[start:end]-noise[start:end])).square().sum())
        return torch.stack(losses).sum()/target.numel()
