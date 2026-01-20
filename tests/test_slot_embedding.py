from __future__ import annotations

import torch

from src.models import HeatmapEncoder


def test_slot_embedding_changes_output() -> None:
    torch.manual_seed(0)
    batch = 1
    steps = 4
    x = torch.rand(batch, steps, 2, 32, 32)
    slot_zero = torch.zeros(batch, steps, dtype=torch.long)
    slot_one = torch.ones(batch, steps, dtype=torch.long)

    encoder = HeatmapEncoder()
    encoder.eval()

    with torch.no_grad():
        z0 = encoder(x, slot_zero)
        z1 = encoder(x, slot_one)

    diff = (z0 - z1).abs().sum().item()
    assert diff > 0
