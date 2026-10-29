"""Task-prefix extension for the simulation-only FPP implementation."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import patch_causal_dit as base


def prefix_causal_mask(q, k, tokens):
    # Prefix is index 0, NOT visual time zero. It cannot read any image.
    return (k == 0) | ((q > 0) & (k > 0) & ((q - 1) // tokens >= (k - 1) // tokens))


class PrefixVisualEncoder(base.PatchTemporalEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.condition = nn.Sequential(nn.LayerNorm(config['text_dim']),
                                       nn.Linear(config['text_dim'], config['visual_dim']))
        self.condition_type = nn.Parameter(torch.zeros(1, 1, config['visual_dim']))

    def _attention(self, steps, device):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        if self._flex is None:
            self._flex = torch.compile(flex_attention, dynamic=True)
        key = (steps, str(device))
        if key not in self._block_masks:
            tokens, length = self.tokens_per_time, 1 + steps * self.tokens_per_time

            def mask(batch, head, q, k):
                return prefix_causal_mask(q, k, tokens)

            self._block_masks[key] = create_block_mask(
                mask, B=None, H=None, Q_LEN=length, KV_LEN=length,
                device=str(device), _compile=True)
        return self._flex, self._block_masks[key]

    def forward(self, features, text):
        batch, steps, tokens, _ = features.shape
        if tokens != self.tokens_per_time or steps > self.max_steps:
            raise ValueError(f'Invalid visual layout: {features.shape}')
        if text.shape != (batch, self.condition[0].normalized_shape[0]):
            raise ValueError(f'Expected one fixed-size text condition per episode: {text.shape}')
        positions = (self.camera[:, None] + self.spatial_position[None]).flatten(0, 1)
        x = self.input_projection(self.input_norm(features))
        x = x + positions.to(x.dtype)[None, None] + self.time_position[:steps].to(x.dtype)[None, :, None]
        prefix = self.condition(text).unsqueeze(1) + self.condition_type
        x = torch.cat((prefix.to(x.dtype), x.flatten(1, 2)), dim=1)
        if x.is_cuda:
            attention, mask = self._attention(steps, x.device)
        else:
            if x.shape[1] > 4096:
                raise ValueError('Long contexts require CUDA FlexAttention')
            indices = torch.arange(x.shape[1], device=x.device)
            mask = prefix_causal_mask(indices[:, None], indices[None], tokens)

            def attention(q, k, v, block_mask):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)

        for block in self.blocks:
            x = checkpoint(block, x, attention, mask, use_reentrant=False) \
                if self.training and self.checkpoint_layers else block(x, attention, mask)
        x = self.output_norm(x)
        prefix = x[:, :1]
        visual = x[:, 1:].reshape(batch, steps, tokens, -1)
        return prefix, visual


class TemporalRoPEVisualBlock(base.VisualBlock):
    """Visual block with temporal-only rotary Q/K positions."""

    def __init__(self, dim, heads, ratio):
        super().__init__(dim, heads, ratio)
        head_dim = dim // heads
        if head_dim % 2:
            raise ValueError('Temporal RoPE requires an even attention head dimension')
        frequency = torch.exp(torch.arange(head_dim // 2).float()
                              * (-math.log(10000.0) / (head_dim // 2)))
        self.register_buffer('rope_frequency', frequency)

    def apply_rope(self, value, temporal_positions):
        angle = temporal_positions.to(self.rope_frequency.dtype)[:, None] * self.rope_frequency[None]
        cosine = angle.cos().to(value.dtype)[None, None]
        sine = angle.sin().to(value.dtype)[None, None]
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack((even*cosine-odd*sine, even*sine+odd*cosine), dim=-1).flatten(-2)

    def forward(self, x, attention, mask, temporal_positions):
        batch, length, dim = x.shape
        q, k, v = self.qkv(self.norm_attention(x)).view(
            batch, length, 3, self.heads, dim // self.heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        q = self.apply_rope(q, temporal_positions)
        k = self.apply_rope(k, temporal_positions)
        update = attention(q, k, v, block_mask=mask)
        x = x + self.projection(update.transpose(1, 2).reshape(batch, length, dim))
        return x + self.ffn(self.norm_ffn(x))


class PrefixTemporalRoPEEncoder(nn.Module):
    """DINO spatial features, learned camera IDs, and temporal RoPE."""

    temporal_rope = True

    def __init__(self, config):
        super().__init__()
        dim = config['visual_dim']
        self.tokens_per_time = config['cameras'] * config['grid_size'] ** 2
        self.max_steps = config['max_steps']
        self.checkpoint_layers = config['checkpoint_visual']
        self.input_norm = nn.LayerNorm(config['feature_dim'])
        self.input_projection = nn.Linear(config['feature_dim'], dim)
        self.camera = nn.Parameter(torch.randn(config['cameras'], dim) * 0.02)
        self.patches_per_camera = config['grid_size'] ** 2
        if config.get('temporal_position') == 'rope_spatial':
            grid = config['grid_size']
            yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
            spatial = torch.cat([base.sinusoidal(yy.flatten(), dim // 2),
                                 base.sinusoidal(xx.flatten(), dim // 2)], dim=-1)
            self.register_buffer('spatial_position', spatial)
        self.blocks = nn.ModuleList([
            TemporalRoPEVisualBlock(dim, config['visual_heads'], config['ffn_ratio'])
            for _ in range(config['visual_layers'])
        ])
        self.output_norm = nn.LayerNorm(dim)
        self.condition = nn.Sequential(nn.LayerNorm(config['text_dim']),
                                       nn.Linear(config['text_dim'], dim))
        self.condition_type = nn.Parameter(torch.zeros(1, 1, dim))
        self._block_masks = {}
        self._flex = None

    def _attention(self, steps, device):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        if self._flex is None:
            self._flex = torch.compile(flex_attention, dynamic=True)
        key = (steps, str(device))
        if key not in self._block_masks:
            tokens, length = self.tokens_per_time, 1 + steps * self.tokens_per_time

            def mask(batch, head, q, k):
                return prefix_causal_mask(q, k, tokens)

            self._block_masks[key] = create_block_mask(
                mask, B=None, H=None, Q_LEN=length, KV_LEN=length,
                device=str(device), _compile=True)
        return self._flex, self._block_masks[key]

    def temporal_positions(self, steps, device, include_prefix=True):
        positions = torch.arange(steps, device=device).repeat_interleave(self.tokens_per_time)
        if include_prefix:
            positions = torch.cat((positions.new_zeros(1), positions))
        return positions

    def forward(self, features, text):
        batch, steps, tokens, _ = features.shape
        if tokens != self.tokens_per_time or steps > self.max_steps:
            raise ValueError(f'Invalid visual layout: {features.shape}')
        if text.shape != (batch, self.condition[0].normalized_shape[0]):
            raise ValueError(f'Expected one fixed-size text condition per episode: {text.shape}')
        camera = self.camera[:, None].expand(-1, self.patches_per_camera, -1).flatten(0, 1)
        if hasattr(self, 'spatial_position'):
            camera = (self.camera[:, None] + self.spatial_position[None]).flatten(0, 1)
        x = self.input_projection(self.input_norm(features))
        x = x + camera.to(x.dtype)[None, None]
        prefix = self.condition(text).unsqueeze(1) + self.condition_type
        x = torch.cat((prefix.to(x.dtype), x.flatten(1, 2)), dim=1)
        temporal_positions = self.temporal_positions(steps, x.device)
        if x.is_cuda:
            attention, mask = self._attention(steps, x.device)
        else:
            if x.shape[1] > 4096:
                raise ValueError('Long contexts require CUDA FlexAttention')
            indices = torch.arange(x.shape[1], device=x.device)
            mask = prefix_causal_mask(indices[:, None], indices[None], tokens)

            def attention(q, k, v, block_mask):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)

        for block in self.blocks:
            x = checkpoint(block, x, attention, mask, temporal_positions, use_reentrant=False) \
                if self.training and self.checkpoint_layers \
                else block(x, attention, mask, temporal_positions)
        x = self.output_norm(x)
        prefix = x[:, :1]
        visual = x[:, 1:].reshape(batch, steps, tokens, -1)
        return prefix, visual


class PrefixPatchDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        temporal_position = config.get('temporal_position', 'absolute')
        if temporal_position == 'absolute':
            self.visual = PrefixVisualEncoder(config)
        elif temporal_position in {'rope', 'rope_spatial'}:
            self.visual = PrefixTemporalRoPEEncoder(config)
        else:
            raise ValueError(f'Unsupported temporal position encoding: {temporal_position}')
        self.action = base.CrossAttentionFlowHead(config)

    def encode(self, features, text):
        prefix, visual = self.visual(features, text)
        return torch.cat((prefix[:, None].expand(-1, visual.shape[1], -1, -1), visual), dim=2)

    def forward(self, features, text, actions, valid, action_microbatch=16):
        if not valid.any():
            raise ValueError('No valid action supervision')
        memory = self.encode(features, text)[valid]
        target = actions[valid].float()
        tau = torch.rand(len(target), device=target.device)
        noise = torch.randn_like(target)
        noisy = (1-tau[:, None, None])*noise + tau[:, None, None]*target
        losses = []
        for start in range(0, len(target), action_microbatch):
            end = start + action_microbatch
            velocity = self.action(noisy[start:end], tau[start:end], memory[start:end])
            losses.append((velocity.float() - target[start:end] + noise[start:end]).square().sum())
        # Text has no target and no independent loss. Its projection receives
        # gradients only through the action objective.
        return torch.stack(losses).sum() / target.numel()
