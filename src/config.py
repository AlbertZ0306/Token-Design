from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HeatmapConfig:
    t_slots: int = 240
    channels: int = 2
    height: int = 32
    width: int = 32
    r_max: float = math.log(1.2)
    s: float = 0.02
    v_cap: int = 50000
    pixel_scale: bool = True
    count_cap: int = 128
    allow_fallback_pref: bool = False


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
