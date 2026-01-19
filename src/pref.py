from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
DATE_RE_COMPACT = re.compile(r"\d{8}")

PREF_NAME_CANDIDATES = ["pref", "close", "prev_close", "preclose", "pre_close"]


def normalize_trade_date(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if DATE_RE.fullmatch(text):
        return text
    if DATE_RE_COMPACT.fullmatch(text):
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d")
    return text


def normalize_stock_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):06d}"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):06d}"
    text = str(value).strip()
    if text.isdigit() and len(text) < 6:
        return text.zfill(6)
    return text


def _detect_pref_column(columns: list[str]) -> str:
    lowered = {c.lower(): c for c in columns}
    for name in PREF_NAME_CANDIDATES:
        if name in lowered:
            return lowered[name]
    raise ValueError("Pref column not found in pref_map")


def _load_pref_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".json", ".jsn"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                return pd.DataFrame(payload["records"])
            rows = []
            for key, value in payload.items():
                if isinstance(key, str) and "," in key:
                    stock_id, trade_date = key.split(",", 1)
                elif isinstance(key, str) and "|" in key:
                    stock_id, trade_date = key.split("|", 1)
                elif isinstance(key, (list, tuple)) and len(key) == 2:
                    stock_id, trade_date = key
                else:
                    continue
                rows.append({"stock_id": stock_id, "trade_date": trade_date, "pref": value})
            return pd.DataFrame(rows)
        raise ValueError("Unsupported JSON format for pref_map")
    raise ValueError(f"Unsupported pref_map file type: {path}")


def load_pref_map(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    if not str(path).strip():
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pref map not found: {path}")

    df = _load_pref_frame(path)
    if df.empty:
        return {}

    columns = [c.strip() for c in df.columns]
    df.columns = columns

    pref_col = _detect_pref_column(columns)
    if "stock_id" not in df.columns:
        raise ValueError("pref_map must include stock_id column")
    if "trade_date" not in df.columns:
        raise ValueError("pref_map must include trade_date column")

    stock_ids = df["stock_id"].map(normalize_stock_id)
    trade_dates = df["trade_date"].map(normalize_trade_date)
    prefs = pd.to_numeric(df[pref_col], errors="coerce")

    pref_map: dict[tuple[str, str], float] = {}
    for stock_id, trade_date, pref in zip(stock_ids, trade_dates, prefs):
        if not stock_id or not trade_date or pd.isna(pref):
            continue
        pref_map[(stock_id, trade_date)] = float(pref)

    return pref_map


def find_pref_candidates(data_dir: Path) -> list[Path]:
    tokens = ["pref", "close", "daily", "kline", "ohlc", "quote", "summary", "meta"]
    exts = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".jsn"}
    candidates: list[Path] = []
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        name = path.name.lower()
        if any(token in name for token in tokens):
            candidates.append(path)
    return candidates
