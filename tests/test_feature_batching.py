import numpy as np
import pytest
import torch

from scripts.prepare_cache import _features


def test_legacy_pairing_preserves_one_view_and_short_final_block() -> None:
    class Encoder:
        num_prefix_tokens = 0

        def __init__(self) -> None:
            self.batches = []

        def forward_features(self, pixels: torch.Tensor) -> torch.Tensor:
            self.batches.append(len(pixels))
            torch.testing.assert_close(pixels[::2], pixels[1::2])
            return pixels.mean((2, 3))[:, None]

    rng = np.random.default_rng(33)
    images = rng.integers(0, 256, (33, 16, 16, 3), dtype=np.uint8)
    model = Encoder()
    actual = _features(model, images, torch.device("cpu"), 64, "legacy-paired32")
    assert model.batches == [64, 2]
    assert actual.shape == (33, 1, 3)
    assert not np.array_equal(actual[0], actual[1])
    with pytest.raises(ValueError, match="batch 64"):
        _features(model, images, torch.device("cpu"), 32, "legacy-paired32")
