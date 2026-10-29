"""A complete visual demo prefix followed by block-causal execution history.

The default head reads the contextualized current frame. Optional ablations add
direct demo attention and learned spatial pooling. BPP-compatible demo and query
modalities can also be combined with the direct path and full causal history.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fpp_libero_mem.patch_causal_dit import CrossAttentionFlowHead, sinusoidal
from fpp_libero_mem.prefix_model import TemporalRoPEVisualBlock
from fpp_libero_mem.patch_causal_dit import Attention
from fpp_libero_gen.direct_head import DirectDemoFlowHead
from fpp_libero_gen.sensorimotor import SensorimotorEmbedding


def demo_causal_mask(q, k, prefix_tokens, frame_tokens):
    return (k < prefix_tokens) | ((q >= prefix_tokens) &
        ((q - prefix_tokens) // frame_tokens >= (k - prefix_tokens) // frame_tokens))


class VideoPrefixEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim, grid = config['visual_dim'], config['grid_size']
        self.query_multimodal = config.get('query_input') == 'bpp'
        self.full_demo_patches = config.get('demo_full_patches', False)
        if self.query_multimodal and config.get('demo_input') != 'bpp':
            raise ValueError('Multimodal query requires multimodal demo encoder')
        if self.full_demo_patches and config.get('demo_input') != 'bpp':
            raise ValueError('Full demo patches require the multimodal demo encoder')
        wrist_tokens = config['demo_grid_size']**2 if config.get('query_wrist_pool') == 'learned' else grid**2
        self.tokens_per_time = grid**2+wrist_tokens+1 if self.query_multimodal else grid**2
        self.max_steps = config['max_steps']
        self.demo_grid = config['demo_grid_size']
        self.input_norm = nn.LayerNorm(config['feature_dim'])
        self.input_projection = nn.Linear(config['feature_dim'], dim)
        self.segment = nn.Parameter(torch.randn(2, dim) * .02)
        for name, size in [('spatial', grid), ('demo_spatial', self.demo_grid)]:
            # Positions share the original camera's coordinate system.
            axis = (torch.arange(size).float() + .5) * grid / size - .5
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
            self.register_buffer(name, torch.cat((sinusoidal(yy.flatten(), dim // 2),
                                                  sinusoidal(xx.flatten(), dim // 2)), -1))
        self.blocks = nn.ModuleList([TemporalRoPEVisualBlock(dim, config['visual_heads'],
            config['ffn_ratio']) for _ in range(config['visual_layers'])])
        self.output_norm = nn.LayerNorm(dim)
        needs_pool = not self.full_demo_patches or (self.query_multimodal and config.get('query_wrist_pool') == 'learned')
        if config.get('demo_pooling', 'average') == 'learned' and needs_pool:
            self.demo_pool_queries = nn.Parameter(torch.randn(self.demo_grid**2, dim) * .02)
            self.demo_pool_attention = Attention(dim, config['visual_heads'])
            self.demo_pool_norm = nn.LayerNorm(dim)
        self.sensorimotor = SensorimotorEmbedding(config) if config.get('demo_input') == 'bpp' else None
        self._masks = {}
        self._flex = None

    @property
    def demo_tokens_per_time(self):
        return self.sensorimotor.tokens_per_chunk if self.sensorimotor is not None else self.demo_grid**2

    def demo_token_mask(self, demo, demo_valid):
        if self.sensorimotor is None:
            if demo_valid is not None:
                raise ValueError('RGB baseline does not use demo padding')
            return None
        if demo_valid is None or demo_valid.shape != demo.shape[:2] or demo_valid.dtype != torch.bool:
            raise ValueError('Sensorimotor demo needs a boolean chunk-valid mask')
        if not demo_valid.any(1).all():
            raise ValueError('Every support demo must have real chunks')
        return demo_valid.repeat_interleave(self.demo_tokens_per_time, dim=1)

    def embed_demo(self, demo, demo_proprio=None, demo_actions=None):
        if self.sensorimotor is not None:
            if demo.shape[1] != self.config['demo_frames'] or demo_proprio is None or demo_actions is None:
                raise ValueError('Missing BPP demo fields/capacity')
            return self.sensorimotor(demo, demo_proprio, demo_actions, self.input_norm,
                self.input_projection, self.spatial if self.full_demo_patches else self.demo_spatial, self.segment[0],
                self.pool_sensorimotor_images if self.config.get('demo_pooling') == 'learned' and not self.full_demo_patches else None)
        b, t, p, d = demo.shape
        grid = self.config['grid_size']
        if p != grid ** 2 or t != self.config['demo_frames']:
            raise ValueError('Demo layout differs from config')
        if self.config.get('demo_pooling', 'average') == 'learned':
            full = self.input_projection(self.input_norm(demo))
            full = full + self.spatial.to(full.dtype)[None, None] + self.segment[0].to(full.dtype)
            queries = self.demo_pool_queries.to(full.dtype) + self.demo_spatial.to(full.dtype)
            queries = queries[None].expand(b*t, -1, -1)
            pooled = queries + self.demo_pool_attention(queries, full.reshape(b*t, p, -1))
            return self.demo_pool_norm(pooled).reshape(b, t*self.demo_grid**2, -1)
        if self.demo_grid != grid:
            demo = F.adaptive_avg_pool2d(demo.reshape(b*t, grid, grid, d).permute(0, 3, 1, 2),
                                        (self.demo_grid, self.demo_grid))
            demo = demo.permute(0, 2, 3, 1).reshape(b, t, self.demo_grid**2, d)
        x = self.input_projection(self.input_norm(demo))
        x = x + self.demo_spatial.to(x.dtype)[None, None] + self.segment[0].to(x.dtype)
        return x.flatten(1, 2)

    def pool_sensorimotor_images(self, demo, views):
        leading = demo.shape[:-3]
        v,p,d = demo.shape[-3:]
        full = self.input_projection(self.input_norm(demo))
        full = full + self.spatial.to(full.dtype) + views.to(full.dtype)[:,None,:]
        queries = self.demo_pool_queries.to(full.dtype) + self.demo_spatial.to(full.dtype)
        queries = queries[None].expand(full.numel()//(p*full.shape[-1]),-1,-1)
        pooled = queries + self.demo_pool_attention(queries, full.reshape(-1,p,full.shape[-1]))
        return self.demo_pool_norm(pooled).reshape(*leading,v,self.demo_grid**2,-1)

    def embed_query(self, features, query_proprio=None):
        if self.query_multimodal:
            if features.shape[-3:-1] != (2, self.config['grid_size']**2) or query_proprio is None:
                raise ValueError('Query requires two camera views and proprio11')
            if self.config.get('query_wrist_pool') == 'learned':
                agent = self.input_projection(self.input_norm(features[...,0,:,:]))
                agent = agent + self.spatial.to(agent.dtype) + self.sensorimotor.view[0].to(agent.dtype)
                wrist = self.pool_sensorimotor_images(features[...,1:2,:,:], self.sensorimotor.view[1:]).squeeze(-3)
                x = torch.cat((agent,wrist), dim=-2)
            else:
                x = self.input_projection(self.input_norm(features))
                x = x + self.spatial.to(x.dtype) + self.sensorimotor.view.to(x.dtype)[:,None,:]
                x = x.flatten(-3,-2)
            prop = self.sensorimotor.proprio(query_proprio) + self.sensorimotor.modality[0].to(x.dtype)
            x = torch.cat((x, prop.unsqueeze(-2)), dim=-2)
            return x + self.segment[1].to(x.dtype)
        x = self.input_projection(self.input_norm(features))
        return x + self.spatial.to(x.dtype) + self.segment[1].to(x.dtype)

    def forward(self, features, demo, return_demo=False, demo_proprio=None, demo_actions=None, demo_valid=None, query_proprio=None):
        embedded = self.embed_query(features, query_proprio)
        b, steps, tokens, _ = embedded.shape
        if tokens != self.tokens_per_time or steps > self.max_steps:
            raise ValueError('Query exceeds configured history/layout')
        prefix = self.embed_demo(demo, demo_proprio, demo_actions)
        prefix_valid = self.demo_token_mask(demo, demo_valid)
        n = prefix.shape[1]
        x = torch.cat((prefix, embedded.flatten(1, 2)), 1)
        positions = torch.cat((torch.arange(demo.shape[1], device=x.device).repeat_interleave(self.demo_tokens_per_time),
            torch.arange(steps, device=x.device).repeat_interleave(tokens)))
        if x.is_cuda:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
            if self._flex is None:
                self._flex = torch.compile(flex_attention, dynamic=True)
            key = (steps, n, str(x.device))
            if prefix_valid is not None:
                # Valid lengths vary per episode. Never reuse a previous batch's padding mask.
                def padded_mask(batch, head, q, k):
                    valid_key = (k >= n) | prefix_valid[batch, k.clamp(max=n-1)]
                    return demo_causal_mask(q, k, n, tokens) & valid_key
                mask = create_block_mask(padded_mask, B=b, H=None, Q_LEN=x.shape[1],
                    KV_LEN=x.shape[1], device=str(x.device), _compile=True)
            elif key not in self._masks:
                def mask(batch, head, q, k):
                    return demo_causal_mask(q, k, n, tokens)
                self._masks[key] = create_block_mask(mask, B=None, H=None,
                    Q_LEN=x.shape[1], KV_LEN=x.shape[1], device=str(x.device), _compile=True)
            if prefix_valid is None:
                mask = self._masks[key]
            attention = self._flex
        else:
            if x.shape[1] > 4096:
                raise ValueError('Long contexts require CUDA FlexAttention')
            indices = torch.arange(x.shape[1], device=x.device)
            mask = demo_causal_mask(indices[:, None], indices[None], n, tokens)
            if prefix_valid is not None:
                valid_keys = torch.cat((prefix_valid, torch.ones((b,steps*tokens),dtype=torch.bool,device=x.device)),1)
                mask = mask[None,None] & valid_keys[:,None,None]
            def attention(q, k, v, block_mask):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)
        for block in self.blocks:
            x = checkpoint(block, x, attention, mask, positions, use_reentrant=False) \
                if self.training and self.config['checkpoint_visual'] \
                else block(x, attention, mask, positions)
        query = self.output_norm(x[:, n:]).reshape(b, steps, tokens, -1)
        return (query, self.output_norm(x[:, :n])) if return_demo else query


class VideoPrefixFPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.direct_demo = config.get('direct_demo', False)
        self.checkpoint_direct = config.get('checkpoint_direct', True)
        self.shared_demo_kv = config.get('shared_demo_kv', False)
        if self.shared_demo_kv and not self.direct_demo:
            raise ValueError('Shared demo K/V requires the direct demo action head')
        self.visual = VideoPrefixEncoder(config)
        self.action = DirectDemoFlowHead(config) if self.direct_demo else CrossAttentionFlowHead(config)

    def forward(self, features, demo, actions, valid, action_microbatch=8,
                demo_proprio=None, demo_actions=None, demo_valid=None, query_proprio=None):
        if not valid.any():
            raise ValueError('No query action targets')
        encoded = self.visual(features, demo, return_demo=self.direct_demo,
            demo_proprio=demo_proprio, demo_actions=demo_actions, demo_valid=demo_valid, query_proprio=query_proprio)
        if self.direct_demo:
            memory, prefix = encoded
            memory = memory[valid]
            owners = valid.nonzero(as_tuple=True)[0]
            prefix_valid = self.visual.demo_token_mask(demo, demo_valid)
            if self.shared_demo_kv:
                demo_kv = checkpoint(self.action.project_demo_kv, prefix, use_reentrant=False) \
                    if self.training and self.checkpoint_direct else self.action.project_demo_kv(prefix)
        else:
            memory = encoded[valid]
        target = actions[valid].float()
        tau = torch.rand(len(target), device=target.device)
        noise = torch.randn_like(target)
        noisy = (1-tau[:, None, None])*noise + tau[:, None, None]*target
        total = memory.new_zeros((), dtype=torch.float32)
        for start in range(0, len(target), action_microbatch):
            stop = start + action_microbatch
            inputs = (noisy[start:stop], tau[start:stop], memory[start:stop])
            if self.direct_demo:
                if self.shared_demo_kv:
                    inputs = (*inputs, prefix)
                    kwargs = dict(demo_valid=prefix_valid, demo_kv=demo_kv,
                                  demo_owners=owners[start:stop])
                else:
                    inputs = (*inputs, prefix[owners[start:stop]])
                    kwargs = {} if prefix_valid is None else dict(demo_valid=prefix_valid[owners[start:stop]])
                velocity = checkpoint(self.action, *inputs, use_reentrant=False, **kwargs) \
                    if self.training and self.checkpoint_direct else self.action(*inputs, **kwargs)
            else:
                velocity = checkpoint(self.action, *inputs, use_reentrant=False) \
                    if self.training and self.visual.config.get('checkpoint_action', False) else self.action(*inputs)
            total = total + (velocity.float() - target[start:stop] + noise[start:stop]).square().sum()
        return total / target.numel()

    def sample_action(self, memory, cache, steps=10, noise=None):
        if self.direct_demo:
            return self.action.sample(memory, cache.demo_memory, steps=steps, noise=noise,
                                      demo_valid=cache.prefix_mask)
        return self.action.sample(memory, steps=steps, noise=noise)


class VideoKVCache:
    """Precompute bidirectional demo K/V, then append complete query frames."""
    def __init__(self, encoder):
        self.encoder = encoder

    @torch.inference_mode()
    def reset(self, demo, demo_proprio=None, demo_actions=None, demo_valid=None):
        e = self.encoder
        if e.training:
            raise RuntimeError('KV cache requires eval mode')
        x = e.embed_demo(demo, demo_proprio, demo_actions)
        self.prefix_mask = e.demo_token_mask(demo, demo_valid)
        self.prefix_length = x.shape[1]
        self.step = 0
        self.layers = []
        positions = torch.arange(demo.shape[1], device=x.device).repeat_interleave(e.demo_tokens_per_time)
        for block in e.blocks:
            q, k, v = self._qkv(block, x, positions)
            shape = (x.shape[0], block.heads, self.prefix_length + e.max_steps*e.tokens_per_time, k.shape[-1])
            kc, vc = torch.empty(shape, device=k.device, dtype=k.dtype), torch.empty(shape, device=v.device, dtype=v.dtype)
            kc[:, :, :self.prefix_length].copy_(k)
            vc[:, :, :self.prefix_length].copy_(v)
            self.layers.append((kc, vc))
            update = F.scaled_dot_product_attention(q, k, v,
                attn_mask=None if self.prefix_mask is None else self.prefix_mask[:,None,None])
            x = x + block.projection(update.transpose(1, 2).reshape_as(x))
            x = x + block.ffn(block.norm_ffn(x))
        self.demo_memory = e.output_norm(x) if e.config.get('direct_demo', False) else None

    @staticmethod
    def _qkv(block, x, positions):
        b, n, d = x.shape
        q, k, v = block.qkv(block.norm_attention(x)).view(b, n, 3, block.heads,
            d // block.heads).permute(2, 0, 3, 1, 4).unbind(0)
        return block.apply_rope(q, positions), block.apply_rope(k, positions), v

    @torch.inference_mode()
    def encode(self, features, query_proprio=None):
        e = self.encoder
        x = e.embed_query(features, query_proprio)
        if self.step >= e.max_steps or x.shape[1] != e.tokens_per_time:
            raise ValueError('History capacity/layout exceeded')
        positions = torch.full((x.shape[1],), self.step, device=x.device)
        begin = self.prefix_length + self.step*e.tokens_per_time
        end = begin + e.tokens_per_time
        for block, (kc, vc) in zip(e.blocks, self.layers):
            q, k, v = self._qkv(block, x, positions)
            kc[:, :, begin:end].copy_(k)
            vc[:, :, begin:end].copy_(v)
            mask = None
            if self.prefix_mask is not None:
                valid = torch.cat((self.prefix_mask, torch.ones((len(x),end-self.prefix_length),device=x.device,dtype=torch.bool)),1)
                mask = valid[:,None,None]
            update = F.scaled_dot_product_attention(q, kc[:, :, :end], vc[:, :, :end], attn_mask=mask)
            x = x + block.projection(update.transpose(1, 2).reshape_as(x))
            x = x + block.ffn(block.norm_ffn(x))
        self.step += 1
        return e.output_norm(x)
