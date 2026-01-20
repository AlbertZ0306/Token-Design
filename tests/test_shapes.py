from __future__ import annotations

import torch

from src.models import HeatmapDecoder, HeatmapEncoder


def test_shapes_and_range() -> None:
    torch.manual_seed(0)
    batch = 2
    steps = 240
    x = torch.rand(batch, steps, 2, 32, 32)
    slot_id = torch.arange(steps).unsqueeze(0).repeat(batch, 1)

    encoder = HeatmapEncoder()
    decoder = HeatmapDecoder()

    z = encoder(x, slot_id)
    assert z.shape == (batch, steps, 4096)

    x_hat = decoder(z)
    assert x_hat.shape == (batch, steps, 2, 32, 32)
    assert x_hat.min() >= -1e-5
    assert x_hat.max() <= 1.0 + 1e-5
