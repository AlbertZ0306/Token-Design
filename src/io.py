from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .config import CANONICAL_COLUMNS, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
YEAR_RE = re.compile(r"\d{4}")
MONTH_DAY_RE = re.compile(r"\d{1,2}")

DEFAULT_SUFFIXES = (".csv", ".tsv", ".txt")


def scan_tick_files(input_dir: Path, suffixes: Sequence[str] = DEFAULT_SUFFIXES) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    files: list[Path] = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)
    return files


def parse_stock_date(path: Path) -> tuple[str, str]:
    stock_id = path.stem
    if not stock_id:
        raise ValueError(f"Empty stock_id from path: {path}")

    trade_date = None
    parts = list(path.parts)
    for part in reversed(parts):
        if DATE_RE.fullmatch(part):
            trade_date = part
            break

    if trade_date is None:
        for i in range(len(parts) - 3, -1, -1):
            year = parts[i]
            if not YEAR_RE.fullmatch(year):
                continue
            month = parts[i + 1]
            day = parts[i + 2]
            if not (MONTH_DAY_RE.fullmatch(month) and MONTH_DAY_RE.fullmatch(day)):
                continue
            trade_date = f"{year}-{int(month):02d}-{int(day):02d}"
            break

    if trade_date is None:
        raise ValueError(f"Unable to parse trade_date from path: {path}")

    return stock_id, trade_date


def _detect_delimiter(sample: str) -> str:
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=[",", "\t"])
        return sniffed.delimiter
    except csv.Error:
        if sample.count("\t") > sample.count(","):
            return "\t"
        return ","


def _normalize_columns(columns: Iterable[object]) -> list[str]:
    lookup = {name.lower(): name for name in CANONICAL_COLUMNS}
    normalized: list[str] = []
    for col in columns:
        col_str = str(col)
        key = col_str.strip().lower()
        normalized.append(lookup.get(key, col_str.strip()))
    return normalized


def _read_with_encoding(path: Path, encoding: str, delimiter: str, header: int | None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=delimiter,
        engine="python",
        encoding=encoding,
        header=header,
    )


def read_tick_file(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, errors="replace") as handle:
                sample = handle.read(4096)
            delimiter = _detect_delimiter(sample)

            df = _read_with_encoding(path, encoding, delimiter, header=0)
            df.columns = _normalize_columns(df.columns)

            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                df = _read_with_encoding(path, encoding, delimiter, header=None)
                if len(df.columns) <= len(CANONICAL_COLUMNS):
                    df.columns = CANONICAL_COLUMNS[: len(df.columns)]
                else:
                    extras = [f"extra_{i}" for i in range(len(df.columns) - len(CANONICAL_COLUMNS))]
                    df.columns = CANONICAL_COLUMNS + extras
                df.columns = _normalize_columns(df.columns)
                missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing required columns {missing} in {path}")

            return df
        except Exception as exc:  # pragma: no cover - only triggers on bad files
            last_error = exc

    raise ValueError(f"Failed to read {path}: {last_error}")
