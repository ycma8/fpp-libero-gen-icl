"""Inference-only cache for the block-causal visual encoder; no new weights."""
import torch
import torch.nn.functional as F


class VisualKVCache:
    def __init__(self, model):
        self.encoder = model.visual
        self.reset()

    def reset(self):
        self.layers = None
        self.prefix = None
        self.step = 0

    @torch.no_grad()
    def encode(self, features, text):
        e = self.encoder
        if e.training:
            raise RuntimeError('KV caching requires eval mode')
        b, tokens, _ = features.shape
        if tokens != e.tokens_per_time or self.step >= e.max_steps:
            raise ValueError('Invalid frame or full history capacity exceeded')
        x = e.input_projection(e.input_norm(features))
        if getattr(e, 'temporal_rope', False):
            camera = e.camera[:, None].expand(-1, e.patches_per_camera, -1).flatten(0, 1)
            if hasattr(e, 'spatial_position'):
                camera = (e.camera[:, None] + e.spatial_position[None]).flatten(0, 1)
            x = x + camera.to(x.dtype)[None]
        else:
            positions = (e.camera[:, None] + e.spatial_position[None]).flatten(0, 1)
            x = x + positions.to(x.dtype)[None] + e.time_position[self.step].to(x.dtype)[None, None]
        first = self.step == 0
        if first:
            prefix = e.condition(text).unsqueeze(1) + e.condition_type
            x = torch.cat((prefix.to(x.dtype), x), 1)
            self.layers = []
        length = 1 + (self.step + 1) * tokens
        for i, block in enumerate(e.blocks):
            q, k, v = block.qkv(block.norm_attention(x)).view(
                b, x.shape[1], 3, block.heads, x.shape[-1] // block.heads
            ).permute(2, 0, 3, 1, 4).unbind(0)
            if getattr(e, 'temporal_rope', False):
                temporal_positions = torch.full(
                    (x.shape[1],), self.step, device=x.device, dtype=torch.long)
                if first:
                    temporal_positions[0] = 0
                q = block.apply_rope(q, temporal_positions)
                k = block.apply_rope(k, temporal_positions)
            if first:
                shape = (b, block.heads, 1 + e.max_steps * tokens, k.shape[-1])
                kc, vc = torch.empty(shape, device=k.device, dtype=k.dtype), torch.empty(shape, device=v.device, dtype=v.dtype)
                self.layers.append((kc, vc))
                begin = 0
            else:
                kc, vc = self.layers[i]
                begin = 1 + self.step * tokens
            kc[:, :, begin:length].copy_(k)
            vc[:, :, begin:length].copy_(v)
            if first:
                # Prefix reads itself only; the whole first frame reads prefix
                # and ALL same-time patches. No triangular patch mask.
                prefix_y = F.scaled_dot_product_attention(q[:, :, :1], k[:, :, :1], v[:, :, :1])
                frame_y = F.scaled_dot_product_attention(q[:, :, 1:], k, v)
                y = torch.cat((prefix_y, frame_y), 2)
            else:
                y = F.scaled_dot_product_attention(q, kc[:, :, :length], vc[:, :, :length])
            x = x + block.projection(y.transpose(1, 2).reshape_as(x))
            x = x + block.ffn(block.norm_ffn(x))
        x = e.output_norm(x)
        if first:
            self.prefix = x[:, :1].clone()
            memory = x
        else:
            memory = torch.cat((self.prefix, x), 1)
        self.step += 1
        return memory
