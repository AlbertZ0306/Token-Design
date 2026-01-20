from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = x + residual
        x = self.act(x)
        return x


class HeatmapEncoder(nn.Module):
    def __init__(
        self,
        normalize: bool = True,
        use_alpha: bool = True,
        use_slot_embedding: bool = True,
        dropout: float = 0.1,
        slot_count: int = 240,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.use_alpha = use_alpha
        self.use_slot_embedding = use_slot_embedding
        self.slot_count = slot_count

        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock(64),
            ResBlock(64),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock(128),
            ResBlock(128),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            ResBlock(256),
            ResBlock(256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 4096),
            nn.LayerNorm(4096),
        )

        if use_alpha:
            self.alpha = nn.Parameter(torch.tensor(0.1))
        else:
            self.alpha = None

        if use_slot_embedding:
            self.slot_embed = nn.Embedding(slot_count, 4096)
            self.slot_norm = nn.LayerNorm(4096)
        else:
            self.slot_embed = None
            self.slot_norm = None

    @staticmethod
    def _normalize_per_image(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        return (x - mean) / (std + 1e-6)

    @staticmethod
    def _flatten_slot_id(
        slot_id: torch.Tensor, expected: int, slot_count: int
    ) -> torch.Tensor:
        if slot_id.dtype != torch.long:
            raise ValueError("slot_id must be torch.long")
        slot_flat = slot_id.reshape(-1)
        if slot_flat.numel() != expected:
            raise ValueError("slot_id size mismatch")
        if slot_flat.min() < 0 or slot_flat.max() >= slot_count:
            raise ValueError("slot_id out of range")
        return slot_flat

    def forward(self, x: torch.Tensor, slot_id: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            batch, steps, channels, height, width = x.shape
            if channels != 2 or height != 32 or width != 32:
                raise ValueError("input must be (B,T,2,32,32)")
            x_reshaped = x.reshape(batch * steps, channels, height, width)
            slot_flat = self._flatten_slot_id(slot_id, batch * steps, self.slot_count)
            out_shape = (batch, steps, 4096)
        elif x.dim() == 4:
            batch = x.shape[0]
            channels, height, width = x.shape[1:]
            if channels != 2 or height != 32 or width != 32:
                raise ValueError("input must be (N,2,32,32)")
            x_reshaped = x
            slot_flat = self._flatten_slot_id(slot_id, batch, self.slot_count)
            out_shape = (batch, 4096)
        else:
            raise ValueError("input must be 4D or 5D")

        if not torch.is_floating_point(x_reshaped):
            x_reshaped = x_reshaped.float()

        if self.normalize:
            x_reshaped = self._normalize_per_image(x_reshaped)

        # (N,2,32,32) -> (N,32,32,32)
        x_reshaped = self.stem(x_reshaped)
        # (N,32,32,32) -> (N,64,16,16)
        x_reshaped = self.down1(x_reshaped)
        # (N,64,16,16) -> (N,128,8,8)
        x_reshaped = self.down2(x_reshaped)
        # (N,128,8,8) -> (N,256,4,4)
        x_reshaped = self.down3(x_reshaped)
        # (N,256,4,4) -> (N,256)
        x_reshaped = self.pool(x_reshaped).reshape(x_reshaped.shape[0], 256)

        z = self.projector(x_reshaped)
        if self.alpha is not None:
            z = self.alpha * z

        if self.slot_embed is not None:
            z = z + self.slot_embed(slot_flat)
            z = self.slot_norm(z)

        return z.reshape(out_shape)
