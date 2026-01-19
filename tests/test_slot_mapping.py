from __future__ import annotations

import pandas as pd

from src.time_binning import map_times_to_slots


def test_slot_mapping_boundaries() -> None:
    times = pd.Series(
        [
            "09:25:00",
            "09:29:00",
            "09:30:00",
            "11:29:00",
            "11:30:00",
            "13:00:00",
            "14:57:00",
            "14:58:00",
            "15:00:00",
            "15:01:00",
        ]
    )
    expected = [0, -1, 1, 120, -1, 121, 238, -1, 239, -1]
    slots = map_times_to_slots(times)
    assert slots.tolist() == expected
