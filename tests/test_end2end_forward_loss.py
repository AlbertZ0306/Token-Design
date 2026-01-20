from __future__ import annotations

import torch

from src.models import AutoRegWrapper


def test_end2end_forward_and_loss() -> None:
    torch.manual_seed(0)
    batch = 2
    steps = 6
    x = torch.rand(batch, steps, 2, 32, 32)
    slot_id = torch.arange(steps).unsqueeze(0).repeat(batch, 1)

    model = AutoRegWrapper()
    x_next, loss = model(x, slot_id)

    assert x_next.shape == (batch, steps, 2, 32, 32)
    assert loss.ndim == 0

    loss.backward()
    assert model.encoder.alpha is not None
    assert model.encoder.alpha.grad is not None
    assert model.decoder.fc.weight.grad is not None
