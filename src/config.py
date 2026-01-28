from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HeatmapConfig:
    t_slots: int = 239
    channels: int = 2
    height: int = 32
    width: int = 32
    r_max: float | None = None  # None表示根据股票代码动态调整
    s: float = 0.02
    v_cap: int = 50000
    pixel_scale: bool = True
    count_cap: int = 128
    allow_fallback_pref: bool = False
    binning_mode: str = "tanh"  # 新增: "tanh" 或 "linear"


CANONICAL_COLUMNS = [
    "TranID",
    "Time",
    "Price",
    "Volume",
    "SaleOrderVolume",
    "BuyOrderVolume",
    "Type",
    "SaleOrderID",
    "SaleOrderPrice",
    "BuyOrderID",
    "BuyOrderPrice",
]

REQUIRED_COLUMNS = ["Time", "Price", "Volume", "Type"]
