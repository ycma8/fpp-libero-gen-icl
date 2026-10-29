import unittest

import torch

from fpp_libero_mem.incremental import VisualKVCache
from fpp_libero_mem.prefix_model import (
    PrefixPatchDiT,
    PrefixTemporalRoPEEncoder,
    PrefixVisualEncoder,
    TemporalRoPEVisualBlock,
    prefix_causal_mask,
)


def small_config():
    return dict(feature_dim=12, text_dim=8, cameras=2, grid_size=2, max_steps=4,
                visual_dim=16, visual_layers=2, visual_heads=2, ffn_ratio=2,
                checkpoint_visual=False, decoder_dim=16, decoder_layers=2,
                decoder_heads=2, action_dim=7, horizon=4)


class PrefixTests(unittest.TestCase):
    def test_incremental_equivalence_and_reset(self):
        for temporal_position in ['absolute','rope','rope_spatial']:
            torch.manual_seed(51)
            config=small_config()
            if temporal_position != 'absolute':
                config['temporal_position']=temporal_position
            m=PrefixPatchDiT(config).eval()
            c=VisualKVCache(m)
            for batch in [1,2]:
                c.reset()
                x,text=torch.randn(batch,4,8,12),torch.randn(batch,8)
                with torch.no_grad():
                    for t in range(4):
                        actual=c.encode(x[:,t],text)
                        expected=m.encode(x[:,:t+1],text)[:,-1]
                        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
                    with self.assertRaises(ValueError):
                        c.encode(x[:,0],text)

    def test_mask(self):
        q = torch.arange(25)
        m = prefix_causal_mask(q[:, None], q[None], 8)
        self.assertEqual(int(m[0].sum()), 1)
        self.assertTrue(m[:, 0].all())
        self.assertTrue(m[1:9, 1:9].all())
        self.assertFalse(m[:9, 9:].any())
        self.assertTrue(m[9:17, :17].all())

    def test_no_future_leakage(self):
        for encoder in [PrefixVisualEncoder, PrefixTemporalRoPEEncoder]:
            torch.manual_seed(3)
            m = encoder(small_config()).eval()
            x, text = torch.randn(1, 3, 8, 12), torch.randn(1, 8)
            p, y = m(x, text)
            changed = x.clone()
            changed[:, 2] += torch.randn_like(changed[:, 2]) * 10
            p2, y2 = m(changed, text)
            torch.testing.assert_close(p, p2)
            torch.testing.assert_close(y[:, :2], y2[:, :2])
            self.assertFalse(torch.allclose(y[:, 2], y2[:, 2]))
            _, y3 = m(x, text + torch.randn_like(text))
            self.assertFalse(torch.allclose(y, y3))

    def test_rope_is_relative_and_spatial_encoding_is_not_duplicated(self):
        torch.manual_seed(8)
        block=TemporalRoPEVisualBlock(16,2,2)
        q,k=torch.randn(2,2,7,8),torch.randn(2,2,7,8)
        positions=torch.arange(7)
        shifted=positions+19
        logits=block.apply_rope(q,positions) @ block.apply_rope(k,positions).transpose(-1,-2)
        shifted_logits=(block.apply_rope(q,shifted)
                        @ block.apply_rope(k,shifted).transpose(-1,-2))
        torch.testing.assert_close(logits,shifted_logits,rtol=2e-5,atol=2e-5)
        encoder=PrefixTemporalRoPEEncoder(small_config())
        self.assertFalse(hasattr(encoder,'spatial_position'))
        self.assertFalse(hasattr(encoder,'time_position'))
        self.assertIsInstance(encoder.camera,torch.nn.Parameter)
        self.assertEqual(encoder.temporal_positions(2,'cpu').tolist(),
                         [0]+[0]*8+[1]*8)

    def test_action_only_gradient_to_prefix(self):
        torch.manual_seed(4)
        m = PrefixPatchDiT(small_config())
        # AdaLN-Zero intentionally gates gradients at initialization; activate it
        # here to test the graph after the first optimizer updates.
        torch.nn.init.normal_(m.action.output_projection.weight, std=.02)
        for block in m.action.blocks:
            torch.nn.init.normal_(block.modulation[-1].weight, std=.02)
        x = torch.randn(1, 3, 8, 12)
        loss = m(x, torch.randn(1, 8), torch.randn(1, 3, 4, 7),
                 torch.tensor([[True, True, False]]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(m.visual.condition[1].weight.grad.abs().sum().item(), 0)



if __name__ == '__main__':
    unittest.main()
