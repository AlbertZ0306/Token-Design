from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io import read_tick_file, scan_tick_files
from src.paths import TICK_DATA_DIR, ANALYSIS_DIR
from src.time_binning import map_times_to_slots, time_to_seconds


T_0925 = 9 * 3600 + 25 * 60
T_0926 = 9 * 3600 + 26 * 60
T_0930 = 9 * 3600 + 30 * 60
T_1130 = 11 * 3600 + 30 * 60
T_1300 = 13 * 3600
T_1458 = 14 * 3600 + 58 * 60
T_1500 = 15 * 3600
T_1501 = 15 * 3600 + 60


def _minute_label(minute: int) -> str:
    hour = minute // 60
    minute = minute % 60
    return f"{hour:02d}:{minute:02d}"


def _bucket_counts(seconds: np.ndarray) -> dict[str, int]:
    buckets = {
        "before_0925": int(np.sum(seconds < T_0925)),
        "between_0926_0930": int(
            np.sum((seconds >= T_0926) & (seconds < T_0930))
        ),
        "between_1130_1300": int(
            np.sum((seconds >= T_1130) & (seconds < T_1300))
        ),
        "between_1458_1500": int(
            np.sum((seconds >= T_1458) & (seconds < T_1500))
        ),
        "after_1501": int(np.sum(seconds >= T_1501)),
    }
    return buckets


def analyze_files(paths: Iterable[Path]) -> dict[str, object]:
    total_ticks = 0
    out_count = 0
    invalid_count = 0
    minute_counts = np.zeros(24 * 60, dtype=np.int64)
    bucket_totals = {
        "before_0925": 0,
        "between_0926_0930": 0,
        "between_1130_1300": 0,
        "between_1458_1500": 0,
        "after_1501": 0,
    }

    for path in paths:
        try:
            df = read_tick_file(path)
        except Exception as exc:
            print(f"Warning: Skipping {path} due to error: {exc}", file=sys.stderr)
            continue

        total_ticks += len(df)

        seconds = time_to_seconds(df["Time"])
        slots = map_times_to_slots(df["Time"])

        invalid_mask = seconds < 0
        invalid_count += int(np.sum(invalid_mask))

        out_mask = (slots < 0) & (seconds >= 0)
        if not np.any(out_mask):
            continue

        out_seconds = seconds[out_mask]
        out_count += int(out_seconds.shape[0])

        minute_idx = (out_seconds // 60).astype(int)
        minute_counts += np.bincount(minute_idx, minlength=24 * 60)

        buckets = _bucket_counts(out_seconds)
        for key, value in buckets.items():
            bucket_totals[key] += value

    return {
        "total_ticks": total_ticks,
        "out_of_session": out_count,
        "invalid_time": invalid_count,
        "minute_counts": minute_counts,
        "bucket_totals": bucket_totals,
    }


def write_csv(path: Path, minute_counts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("minute,count\n")
        for idx, count in enumerate(minute_counts.tolist()):
            handle.write(f"{_minute_label(idx)},{count}\n")


def print_summary(summary: dict[str, object], top_k: int) -> None:
    total_ticks = summary["total_ticks"]
    out_count = summary["out_of_session"]
    invalid = summary["invalid_time"]
    bucket_totals = summary["bucket_totals"]
    minute_counts = summary["minute_counts"]

    print(f"total_ticks={total_ticks}")
    print(f"out_of_session={out_count}")
    print(f"invalid_time={invalid}")

    if out_count > 0:
        print("bucket_distribution:")
        for key, value in bucket_totals.items():
            ratio = value / out_count
            print(f"  {key}: {value} ({ratio:.2%})")
        other = out_count - sum(bucket_totals.values())
        if other:
            ratio = other / out_count
            print(f"  other: {other} ({ratio:.2%})")

    top_k = max(int(top_k), 1)
    if out_count > 0:
        print("top_minutes:")
        top_idx = np.argsort(minute_counts)[::-1][:top_k]
        for idx in top_idx:
            count = int(minute_counts[idx])
            if count == 0:
                break
            print(f"  {_minute_label(int(idx))}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze out-of-session ticks")
    parser.add_argument(
        "--input",
        default=str(TICK_DATA_DIR),
        help="Input data directory",
    )
    parser.add_argument(
        "--output_csv",
        default=str(ANALYSIS_DIR / "minute_distribution.csv"),
        help="Optional CSV output path for minute distribution",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_files", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = scan_tick_files(Path(args.input))
    if args.max_files:
        files = files[: args.max_files]

    summary = analyze_files(files)
    print_summary(summary, args.top_k)

    if args.output_csv:
        write_csv(Path(args.output_csv), summary["minute_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
