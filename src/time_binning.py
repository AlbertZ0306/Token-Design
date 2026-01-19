from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_SLOT_COUNT = 240

START_0925 = 9 * 3600 + 25 * 60
END_0926 = 9 * 3600 + 26 * 60
START_0930 = 9 * 3600 + 30 * 60
END_1130 = 11 * 3600 + 30 * 60
START_1300 = 13 * 3600
END_1458 = 14 * 3600 + 58 * 60
START_1500 = 15 * 3600
END_1501 = 15 * 3600 + 60


def time_to_seconds(time_series: pd.Series) -> np.ndarray:
    extracted = time_series.astype(str).str.extract(
        r"(?P<h>\d{1,2}):(?P<m>\d{1,2}):(?P<s>\d{1,2})"
    )
    hours = pd.to_numeric(extracted["h"], errors="coerce").fillna(-1).astype(int)
    minutes = pd.to_numeric(extracted["m"], errors="coerce").fillna(-1).astype(int)
    seconds = pd.to_numeric(extracted["s"], errors="coerce").fillna(-1).astype(int)

    total_seconds = hours * 3600 + minutes * 60 + seconds
    invalid = (hours < 0) | (minutes < 0) | (seconds < 0)
    total_seconds = total_seconds.to_numpy()
    total_seconds[invalid.to_numpy()] = -1
    return total_seconds


def map_times_to_slots(time_series: pd.Series) -> np.ndarray:
    sec = time_to_seconds(time_series)
    slots = np.full(sec.shape[0], -1, dtype=np.int32)

    mask = (sec >= START_0925) & (sec < END_0926)
    slots[mask] = 0

    mask = (sec >= START_0930) & (sec < END_1130)
    slots[mask] = 1 + ((sec[mask] - START_0930) // 60)

    mask = (sec >= START_1300) & (sec < END_1458)
    slots[mask] = 121 + ((sec[mask] - START_1300) // 60)

    mask = (sec >= START_1500) & (sec < END_1501)
    slots[mask] = 239

    return slots
