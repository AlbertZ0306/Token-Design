from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import HeatmapConfig
from src.pipeline import run_pipeline


def _str_to_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tick->minute heatmaps")
    parser.add_argument(
        "--input",
        default="/home/chenyongyuan/tick_tokenizer/data",
        help="Input data directory",
    )
    parser.add_argument("--output", default="out", help="Output directory")
    parser.add_argument("--pref_map", default="", help="Pref map path (CSV/Parquet/JSON)")
    parser.add_argument("--auto_pref", type=_str_to_bool, default=False, help="Auto-detect pref map")
    parser.add_argument("--pixel_scale", type=_str_to_bool, default=True)
    parser.add_argument("--count_cap", type=int, default=128)
    parser.add_argument("--r_max", type=float, default=math.log(1.2))
    parser.add_argument("--s", type=float, default=0.02)
    parser.add_argument("--v_cap", type=int, default=50000)
    parser.add_argument("--allow_fallback_pref", type=_str_to_bool, default=False)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", type=_str_to_bool, default=True, help="Show progress bar")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pref_map_path = Path(args.pref_map) if args.pref_map else None
    config = HeatmapConfig(
        pixel_scale=args.pixel_scale,
        count_cap=args.count_cap,
        r_max=args.r_max,
        s=args.s,
        v_cap=args.v_cap,
        allow_fallback_pref=args.allow_fallback_pref,
    )

    run_pipeline(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        pref_map_path=pref_map_path,
        config=config,
        workers=args.workers,
        auto_pref=args.auto_pref,
        show_progress=args.progress,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
