from __future__ import annotations

from pathlib import Path

import pytest

from src.io import parse_stock_date


def test_parse_stock_date_with_cn_year(tmp_path: Path) -> None:
    test_path = tmp_path / "2025年" / "202501" / "2025-01-09" / "601599.csv"
    test_path.parent.mkdir(parents=True)
    stock_id, trade_date = parse_stock_date(test_path)
    assert stock_id == "601599"
    assert trade_date == "2025-01-09"


def test_parse_stock_date_with_compact_date(tmp_path: Path) -> None:
    test_path = tmp_path / "2025" / "01" / "09" / "20250109" / "000001.csv"
    test_path.parent.mkdir(parents=True)
    stock_id, trade_date = parse_stock_date(test_path)
    assert stock_id == "000001"
    assert trade_date == "2025-01-09"
