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
            "14:56:00",
            "14:57:00",
            "15:00:00",
            "15:01:00",
        ]
    )
    # New slot boundaries:
    # [09:25:00, 09:25:59) -> slot 0
    # [09:30:00, 11:29:59) -> slots 1-120 (120 minutes)
    # [13:00:00, 14:56:59) -> slots 121-237 (117 minutes)
    # [15:00:00, 15:00:59) -> slot 238
    expected = [0, -1, 1, 120, -1, 121, 237, -1, 238, -1]
    slots = map_times_to_slots(times)
    assert slots.tolist() == expected
