from __future__ import annotations

import torch
from torch import nn

from .heatmap_decoder import HeatmapDecoder
from .heatmap_encoder import HeatmapEncoder


class AutoRegWrapper(nn.Module):
    def __init__(
        self,
        encoder: HeatmapEncoder | None = None,
        decoder: HeatmapDecoder | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else HeatmapEncoder()
        self.decoder = decoder if decoder is not None else HeatmapDecoder()

        # Placeholder causal module: time-independent linear block.
        self.dummy_causal = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096),
        )
        self.head = nn.Linear(4096, 4096)
        self.loss_fn = nn.SmoothL1Loss()

    def forward(
        self, x: torch.Tensor, slot_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 5:
            raise ValueError("input must be (B,T,2,32,32)")

        z_in = self.encoder(x, slot_id)
        if z_in.dim() != 3:
            raise ValueError("encoder output must be (B,T,4096)")

        h = self.dummy_causal(z_in)
        z_next = self.head(h)
        x_next = self.decoder(z_next)

        if x_next.shape[:2] != x.shape[:2]:
            raise ValueError("decoder output shape mismatch")
        if x.shape[1] < 2:
            raise ValueError("sequence length must be >= 2")

        loss = self.loss_fn(x_next[:, :-1], x[:, 1:])
        return x_next, loss
