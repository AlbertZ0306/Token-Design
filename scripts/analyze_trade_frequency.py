from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io import scan_tick_files, read_tick_file, parse_stock_date
from src.paths import TICK_DATA_DIR, ANALYSIS_DIR


logger = logging.getLogger(__name__)


def _parse_time_to_minutes(time_str: str) -> int:
    """将时间字符串 HH:MM:SS 转换为从00:00:00开始的分钟数"""
    try:
        parts = str(time_str).strip().split(":")
        if len(parts) >= 2:
            hour = int(parts[0])
            minute = int(parts[1])
            return hour * 60 + minute
        return None
    except (ValueError, AttributeError):
        return None


def analyze_single_file(path: Path) -> dict[str, Any]:
    """分析单个文件的每分钟交易数量"""
    try:
        df = read_tick_file(path)

        if df.empty or "Time" not in df.columns:
            return {"error": "No data or Time column", "path": str(path)}

        # 解析时间到分钟
        df["_minute"] = df["Time"].apply(_parse_time_to_minutes)
        df = df[df["_minute"].notna()]

        if df.empty:
            return {"error": "No valid times", "path": str(path)}

        # 按分钟统计交易数量
        minute_counts = df.groupby("_minute").size()

        stats = {
            "path": str(path),
            "total_trades": len(df),
            "total_minutes": len(minute_counts),
            "mean_trades_per_minute": float(minute_counts.mean()),
            "std_trades_per_minute": float(minute_counts.std()),
            "min_trades_per_minute": int(minute_counts.min()),
            "max_trades_per_minute": int(minute_counts.max()),
            "p50_trades_per_minute": float(minute_counts.quantile(0.50)),
            "p75_trades_per_minute": float(minute_counts.quantile(0.75)),
            "p90_trades_per_minute": float(minute_counts.quantile(0.90)),
            "p95_trades_per_minute": float(minute_counts.quantile(0.95)),
            "p99_trades_per_minute": float(minute_counts.quantile(0.99)),
            "p999_trades_per_minute": float(minute_counts.quantile(0.999)),
        }

        return stats

    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        return {"error": str(e), "path": str(path)}


def analyze_directory(input_dir: Path, output_file: Path | None = None, show_progress: bool = True) -> pd.DataFrame:
    """分析目录下所有文件的每分钟交易数量统计"""
    files = scan_tick_files(input_dir)

    if not files:
        logger.warning(f"No files found in {input_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(files)} files to analyze")

    results = []
    errors = []

    iterator = tqdm(files, desc="Analyzing files") if show_progress else files

    for path in iterator:
        stock_id, trade_date = parse_stock_date(path)
        stats = analyze_single_file(path)

        if "error" in stats:
            errors.append(stats)
        else:
            stats["stock_id"] = stock_id
            stats["trade_date"] = trade_date
            results.append(stats)

    if not results:
        logger.error("No valid results from any files")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 计算全局统计（跨所有文件的聚合统计）
    all_stats = {
        "total_files": len(results),
        "total_files_with_errors": len(errors),
        "mean_of_mean_trades": float(df["mean_trades_per_minute"].mean()),
        "mean_of_p95_trades": float(df["p95_trades_per_minute"].mean()),
        "mean_of_p99_trades": float(df["p99_trades_per_minute"].mean()),
        "mean_of_p999_trades": float(df["p999_trades_per_minute"].mean()),
        "std_of_mean_trades": float(df["mean_trades_per_minute"].std()),
        "min_of_mean_trades": float(df["mean_trades_per_minute"].min()),
        "max_of_mean_trades": float(df["mean_trades_per_minute"].max()),
    }

    # 打印统计结果
    print("\n" + "=" * 60)
    print("每分钟交易数量统计 (跨所有文件聚合)")
    print("=" * 60)
    print(f"分析文件数量: {all_stats['total_files']}")
    print(f"错误文件数量: {all_stats['total_files_with_errors']}")
    print()
    print("指标 (按文件平均后):")
    print(f"  平均每分钟交易数: {all_stats['mean_of_mean_trades']:.2f}")
    print(f"  平均95分位数:     {all_stats['mean_of_p95_trades']:.2f}")
    print(f"  平均99分位数:     {all_stats['mean_of_p99_trades']:.2f}")
    print(f"  平均99.9分位数:   {all_stats['mean_of_p999_trades']:.2f}")
    print()
    print("分布统计 (平均值的分布):")
    print(f"  标准差:           {all_stats['std_of_mean_trades']:.2f}")
    print(f"  最小值:           {all_stats['min_of_mean_trades']:.2f}")
    print(f"  最大值:           {all_stats['max_of_mean_trades']:.2f}")
    print("=" * 60)

    # 详细分位数表格
    print("\n各文件指标的分位数分布:")
    print("-" * 60)
    for metric in ["mean", "p95", "p99", "p999"]:
        col = f"{metric}_trades_per_minute"
        print(f"\n{metric.upper()} 每分钟交易数的分位数:")
        print(f"  50% (中位数): {df[col].quantile(0.50):.2f}")
        print(f"  75%:          {df[col].quantile(0.75):.2f}")
        print(f"  90%:          {df[col].quantile(0.90):.2f}")
        print(f"  95%:          {df[col].quantile(0.95):.2f}")
        print(f"  99%:          {df[col].quantile(0.99):.2f}")
    print("-" * 60)

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

    return df


def _str_to_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计每分钟交易数量分布")
    parser.add_argument(
        "--input",
        default=str(TICK_DATA_DIR),
        help="输入数据目录",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出结果CSV文件路径 (可选)",
    )
    parser.add_argument(
        "--progress",
        type=_str_to_bool,
        default=True,
        help="显示进度条",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    analyze_directory(
        input_dir=Path(args.input),
        output_file=Path(args.output) if args.output else None,
        show_progress=args.progress,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
