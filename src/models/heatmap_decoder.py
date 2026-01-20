from __future__ import annotations

import torch
from torch import nn


class HeatmapDecoder(nn.Module):
    def __init__(self, upsample_mode: str = "nearest") -> None:
        super().__init__()
        self.upsample_mode = upsample_mode

        self.fc = nn.Linear(4096, 256 * 4 * 4)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.out_conv = nn.Conv2d(32, 2, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            batch, steps, hidden = z.shape
            if hidden != 4096:
                raise ValueError("input must be (B,T,4096)")
            z_flat = z.reshape(batch * steps, hidden)
            out_shape = (batch, steps, 2, 32, 32)
        elif z.dim() == 2:
            batch, hidden = z.shape
            if hidden != 4096:
                raise ValueError("input must be (N,4096)")
            z_flat = z
            out_shape = (batch, 2, 32, 32)
        else:
            raise ValueError("input must be 2D or 3D")

        if not torch.is_floating_point(z_flat):
            z_flat = z_flat.float()

        # (N,4096) -> (N,256*4*4)
        x = self.fc(z_flat)
        # (N,256*4*4) -> (N,256,4,4)
        x = x.reshape(z_flat.shape[0], 256, 4, 4)
        # (N,256,4,4) -> (N,128,8,8)
        x = self.up1(x)
        # (N,128,8,8) -> (N,64,16,16)
        x = self.up2(x)
        # (N,64,16,16) -> (N,32,32,32)
        x = self.up3(x)
        # (N,32,32,32) -> (N,2,32,32)
        x = self.out_conv(x)
        x = self.out_act(x)

        return x.reshape(out_shape)
