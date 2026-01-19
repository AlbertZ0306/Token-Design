from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CANONICAL_COLUMNS, HeatmapConfig
from src.pipeline import process_file


def test_end2end_small(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "2018" / "12" / "25" / "2018-12-25"
    input_dir.mkdir(parents=True)
    file_path = input_dir / "000001.csv"

    df = pd.DataFrame(
        [
            [1, "09:25:00", 10.0, 0, 0, 0, "B", 1, 0, 1, 0],
            [2, "09:30:00", 12.0, 50000, 0, 0, "S", 1, 0, 1, 0],
            [3, "14:57:00", 10.0 / 1.2, 60000, 0, 0, "B", 1, 0, 1, 0],
            [4, "15:00:00", 10.0, 1, 0, 0, "X", 1, 0, 1, 0],
        ],
        columns=CANONICAL_COLUMNS,
    )
    df.to_csv(file_path, index=False)

    output_dir = tmp_path / "out"
    pref_map = {("000001", "2018-12-25"): 10.0}
    config = HeatmapConfig(pixel_scale=True)

    result = process_file(file_path, output_dir, pref_map, config)
    assert result.status == "ok"
    assert result.output_path is not None

    data = np.load(result.output_path, allow_pickle=True)
    heatmaps = data["heatmaps"]
    metadata = json.loads(data["metadata"].item())

    assert heatmaps.shape == (240, 2, 32, 32)
    assert np.all(heatmaps[2] == 0)

    expected = math.log1p(1) / math.log1p(config.count_cap)
    assert np.isclose(heatmaps[0, 0, 0, 15], expected, atol=1e-3)
    assert np.isclose(heatmaps[1, 1, 31, 31], expected, atol=1e-3)
    assert np.isclose(heatmaps[238, 0, 31, 0], expected, atol=1e-3)

    assert metadata["unknown_type_count"] == 1
    assert metadata["out_of_session_count"] == 0
