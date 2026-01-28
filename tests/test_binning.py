from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import HeatmapConfig
from src.heatmap import build_heatmaps, price_to_x_idx, price_to_x_idx_linear, volume_to_y_idx


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
    config = HeatmapConfig(pixel_scale=False, r_max=math.log(1.1))
    heatmaps, stats = build_heatmaps(df, pref=10.0, config=config)

    assert stats.unknown_type_count == 1
    assert heatmaps[0, 0, 0, 15] == 1
    assert heatmaps[0, 1, 0, 15] == 1


def test_linear_binning_edges_main_board() -> None:
    """测试主板（10%限制）线性分桶边界"""
    pref = 10.0
    r_max = math.log(1.1)
    # 使用精确的边界值 exp(-r_max) 和 exp(r_max)
    prices = np.array([pref * math.exp(-r_max), pref, pref * math.exp(r_max)])
    x_idx = price_to_x_idx_linear(prices, pref, width=32, r_max=r_max)
    # Lower bound -> 0, middle value -> 14 (not exactly center), upper bound -> 31
    assert x_idx[0] == 0
    assert x_idx[-1] == 31


def test_linear_binning_edges_gem() -> None:
    """测试创业板（20%限制）线性分桶边界"""
    pref = 10.0
    r_max = math.log(1.2)
    # 使用精确的边界值 exp(-r_max) 和 exp(r_max)
    prices = np.array([pref * math.exp(-r_max), pref, pref * math.exp(r_max)])
    x_idx = price_to_x_idx_linear(prices, pref, width=32, r_max=r_max)
    # Lower bound -> 0, middle value -> 14 (not exactly center), upper bound -> 31
    assert x_idx[0] == 0
    assert x_idx[-1] == 31


def test_linear_binning_clipping() -> None:
    """测试越界值裁断"""
    pref = 10.0
    prices = np.array([pref * 0.5, pref * 0.85, pref * 1.15, pref * 2.0])
    x_idx = price_to_x_idx_linear(prices, pref, width=32, r_max=math.log(1.1))
    assert x_idx[0] == 0  # 裁断到下界
    assert x_idx[-1] == 31  # 裁断到上界


def test_linear_binning_uniform_spacing() -> None:
    """测试线性分桶的均匀分布特性"""
    pref = 10.0
    r_max = math.log(1.1)
    ratio_lower = math.exp(-r_max)
    ratio_upper = math.exp(r_max)

    ratios = np.linspace(ratio_lower, ratio_upper, 5)
    prices = ratios * pref

    x_idx = price_to_x_idx_linear(prices, pref, width=32, r_max=r_max)
    expected = np.linspace(0, 31, 5).astype(np.int32)
    assert np.allclose(x_idx, expected, atol=1)


def test_linear_vs_tanh_difference() -> None:
    """测试两种模式输出不同"""
    pref = 10.0
    prices = np.array([pref * 0.95, pref, pref * 1.05])
    r_max = math.log(1.1)

    x_idx_linear = price_to_x_idx_linear(prices, pref, width=32, r_max=r_max)
    x_idx_tanh = price_to_x_idx(prices, pref, width=32, r_max=r_max, s=0.02)

    assert not np.array_equal(x_idx_linear, x_idx_tanh)
