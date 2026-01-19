from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import HeatmapConfig
from .time_binning import map_times_to_slots


@dataclass(frozen=True)
class HeatmapStats:
    total_ticks: int
    out_of_session_count: int
    unknown_type_count: int


def price_to_x_idx(prices: np.ndarray, pref: float, width: int, r_max: float, s: float) -> np.ndarray:
    if pref <= 0:
        raise ValueError("Pref must be positive")
    if width <= 1:
        raise ValueError("Width must be > 1")
    if s <= 0:
        raise ValueError("s must be positive")

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(prices / pref)

    r_clipped = np.clip(r, -r_max, r_max)
    denom = math.tanh(r_max / s)
    if denom == 0:
        raise ValueError("Invalid r_max/s for tanh denominator")

    x = np.tanh(r_clipped / s) / denom
    x_idx = np.floor(((x + 1.0) / 2.0) * (width - 1)).astype(np.int32)
    return np.clip(x_idx, 0, width - 1)


def volume_to_y_idx(volumes: np.ndarray, height: int, v_cap: int) -> np.ndarray:
    if height <= 1:
        raise ValueError("Height must be > 1")
    if v_cap <= 0:
        raise ValueError("v_cap must be positive")

    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.log1p(volumes)
    u_max = math.log1p(v_cap)
    y_idx = np.floor((np.minimum(u, u_max) / u_max) * (height - 1)).astype(np.int32)
    return np.clip(y_idx, 0, height - 1)


def apply_pixel_scale(heatmaps: np.ndarray, count_cap: int, dtype: np.dtype) -> np.ndarray:
    if count_cap <= 0:
        raise ValueError("count_cap must be positive")
    cap = math.log1p(count_cap)
    scaled = np.log1p(heatmaps.astype(np.float32))
    scaled = np.minimum(scaled, cap) / cap
    return scaled.astype(dtype)


def build_heatmaps(df: pd.DataFrame, pref: float, config: HeatmapConfig) -> tuple[np.ndarray, HeatmapStats]:
    total_ticks = int(len(df))

    slots = map_times_to_slots(df["Time"])
    out_of_session_count = int(np.sum(slots < 0))

    prices = pd.to_numeric(df["Price"], errors="coerce").to_numpy()
    volumes = pd.to_numeric(df["Volume"], errors="coerce").to_numpy()
    types = df["Type"].astype(str).str.strip().str.upper().to_numpy()

    valid_mask = slots >= 0
    valid_mask &= np.isfinite(prices) & (prices > 0)
    valid_mask &= np.isfinite(volumes) & (volumes >= 0)

    heatmaps = np.zeros(
        (config.t_slots, config.channels, config.height, config.width), dtype=np.int32
    )

    unknown_type_count = 0
    if np.any(valid_mask):
        slot_idx = slots[valid_mask]
        x_idx = price_to_x_idx(prices[valid_mask], pref, config.width, config.r_max, config.s)
        y_idx = volume_to_y_idx(volumes[valid_mask], config.height, config.v_cap)
        types_valid = types[valid_mask]

        buy_mask = types_valid == "B"
        sell_mask = types_valid == "S"
        unknown_type_count = int(np.sum(~(buy_mask | sell_mask)))

        if np.any(buy_mask):
            np.add.at(heatmaps, (slot_idx[buy_mask], 0, y_idx[buy_mask], x_idx[buy_mask]), 1)
        if np.any(sell_mask):
            np.add.at(heatmaps, (slot_idx[sell_mask], 1, y_idx[sell_mask], x_idx[sell_mask]), 1)

    if config.pixel_scale:
        heatmaps = apply_pixel_scale(heatmaps, config.count_cap, np.float16)

    stats = HeatmapStats(
        total_ticks=total_ticks,
        out_of_session_count=out_of_session_count,
        unknown_type_count=unknown_type_count,
    )
    return heatmaps, stats
