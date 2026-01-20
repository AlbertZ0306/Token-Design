from __future__ import annotations

from pathlib import Path

from src.io import parse_stock_date


def test_parse_stock_date_with_cn_year() -> None:
    path = Path(
        "/home/chenyongyuan/tick_tokenizer/data/2025年/202501/2025-01-09/601599.csv"
    )
    stock_id, trade_date = parse_stock_date(path)
    assert stock_id == "601599"
    assert trade_date == "2025-01-09"


def test_parse_stock_date_with_compact_date() -> None:
    path = Path("/data/2025/01/09/20250109/000001.csv")
    stock_id, trade_date = parse_stock_date(path)
    assert stock_id == "000001"
    assert trade_date == "2025-01-09"
