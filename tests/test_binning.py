from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import HeatmapConfig
from src.heatmap import build_heatmaps, price_to_x_idx, volume_to_y_idx


def test_price_binning_edges() -> None:
    pref = 10.0
    prices = np.array([pref, pref * 1.2, pref / 1.2])
    x_idx = price_to_x_idx(prices, pref, width=32, r_max=math.log(1.2), s=0.02)
    assert x_idx.tolist() == [15, 31, 0]


def test_volume_binning_edges() -> None:
    volumes = np.array([0, 50000, 60000])
    y_idx = volume_to_y_idx(volumes, height=32, v_cap=50000)
    assert y_idx.tolist() == [0, 31, 31]


def test_unknown_type_count() -> None:
    df = pd.DataFrame(
        {
            "Time": ["09:25:00", "09:25:00", "09:25:00"],
            "Price": [10.0, 10.0, 10.0],
            "Volume": [0, 0, 0],
            "Type": ["B", "S", "X"],
        }
    )
    config = HeatmapConfig(pixel_scale=False)
    heatmaps, stats = build_heatmaps(df, pref=10.0, config=config)

    assert stats.unknown_type_count == 1
    assert heatmaps[0, 0, 0, 15] == 1
    assert heatmaps[0, 1, 0, 15] == 1
