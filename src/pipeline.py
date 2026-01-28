from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd

from .config import HeatmapConfig
from .heatmap import build_heatmaps
from .io import parse_stock_date, read_tick_file, scan_tick_files
from .pref import (
    find_pref_candidates,
    load_pref_map,
    normalize_stock_id,
    normalize_trade_date,
)
from .stock_board import get_r_max_for_stock

logger = logging.getLogger(__name__)


class ProgressBar:
    def __init__(self, total: int, enabled: bool = True, width: int = 28) -> None:
        self.total = max(int(total), 0)
        self.enabled = enabled
        self.width = max(int(width), 10)
        self.current = 0
        self.start_time = time.time()
        self._last_render = 0.0
        self._last_len = 0

    def update(self, step: int = 1) -> None:
        if not self.enabled:
            return
        self.current += step
        now = time.time()
        if now - self._last_render >= 0.1 or self.current >= self.total:
            self._render(final=self.current >= self.total)
            self._last_render = now

    def _render(self, final: bool = False) -> None:
        if not self.enabled:
            return
        if self.total > 0:
            ratio = min(max(self.current / self.total, 0.0), 1.0)
        else:
            ratio = 1.0 if final else 0.0
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.time() - self.start_time, 1e-6)
        rate = self.current / elapsed
        msg = f"\r[{bar}] {self.current}/{self.total} {ratio * 100:5.1f}% {rate:6.1f}/s"
        pad = max(self._last_len - len(msg), 0)
        sys.stderr.write(msg + (" " * pad))
        sys.stderr.flush()
        self._last_len = len(msg)
        if final:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def finish(self) -> None:
        if self.enabled:
            self._render(final=True)


@dataclass(frozen=True)
class ProcessResult:
    status: str
    stock_id: str
    trade_date: str
    output_path: str | None
    total_ticks: int
    out_of_session_count: int
    unknown_type_count: int
    reason: str | None = None
    used_fallback_pref: bool = False


def _resolve_pref(
    df: pd.DataFrame,
    pref_map: dict[tuple[str, str], float],
    stock_id: str,
    trade_date: str,
    allow_fallback_pref: bool,
) -> tuple[float | None, bool]:
    key = (normalize_stock_id(stock_id), normalize_trade_date(trade_date))
    if key in pref_map:
        return pref_map[key], False

    if not allow_fallback_pref:
        return None, False

    prices = pd.to_numeric(df["Price"], errors="coerce")
    valid_prices = prices[prices > 0]
    if valid_prices.empty:
        return None, False
    return float(valid_prices.iloc[0]), True


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch is required for tensor output") from exc
    return torch


def _sanitize_metadata(metadata: dict[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, np.generic):
            sanitized[key] = value.item()
        else:
            sanitized[key] = value
    return sanitized


def save_heatmaps(path: Path, heatmaps: np.ndarray, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch = _require_torch()
    payload = {
        "heatmaps": torch.from_numpy(heatmaps),
        "metadata": _sanitize_metadata(metadata),
    }
    torch.save(payload, path)


def process_file(
    path: Path,
    output_dir: Path,
    pref_map: dict[tuple[str, str], float],
    config: HeatmapConfig,
) -> ProcessResult:
    try:
        stock_id, trade_date = parse_stock_date(path)
    except Exception as exc:
        return ProcessResult(
            status="error",
            stock_id="",
            trade_date="",
            output_path=None,
            total_ticks=0,
            out_of_session_count=0,
            unknown_type_count=0,
            reason=f"parse_error: {exc}",
        )

    try:
        df = read_tick_file(path)
    except Exception as exc:
        return ProcessResult(
            status="error",
            stock_id=stock_id,
            trade_date=trade_date,
            output_path=None,
            total_ticks=0,
            out_of_session_count=0,
            unknown_type_count=0,
            reason=f"read_error: {exc}",
        )
    pref, used_fallback = _resolve_pref(
        df, pref_map, stock_id, trade_date, config.allow_fallback_pref
    )
    if pref is None:
        return ProcessResult(
            status="skipped",
            stock_id=stock_id,
            trade_date=trade_date,
            output_path=None,
            total_ticks=int(len(df)),
            out_of_session_count=0,
            unknown_type_count=0,
            reason="pref_missing",
        )
    if pref <= 0:
        return ProcessResult(
            status="error",
            stock_id=stock_id,
            trade_date=trade_date,
            output_path=None,
            total_ticks=int(len(df)),
            out_of_session_count=0,
            unknown_type_count=0,
            reason="pref_invalid",
        )

    # 动态计算r_max
    if config.r_max is None:
        dynamic_r_max = get_r_max_for_stock(stock_id)
    else:
        dynamic_r_max = config.r_max

    # 创建临时配置覆盖r_max
    config_with_r_max = replace(config, r_max=dynamic_r_max)

    try:
        heatmaps, stats = build_heatmaps(df, pref, config_with_r_max)
    except Exception as exc:
        return ProcessResult(
            status="error",
            stock_id=stock_id,
            trade_date=trade_date,
            output_path=None,
            total_ticks=int(len(df)),
            out_of_session_count=0,
            unknown_type_count=0,
            reason=f"build_error: {exc}",
        )

    metadata = {
        "stock_id": stock_id,
        "trade_date": trade_date,
        "pref": pref,
        "pixel_scale": config_with_r_max.pixel_scale,
        "count_cap": config_with_r_max.count_cap,
        "r_max": config_with_r_max.r_max,
        "s": config_with_r_max.s,
        "binning_mode": config_with_r_max.binning_mode,
        "v_cap": config_with_r_max.v_cap,
        "t_slots": config_with_r_max.t_slots,
        "channels": config_with_r_max.channels,
        "height": config_with_r_max.height,
        "width": config_with_r_max.width,
        "unknown_type_count": stats.unknown_type_count,
        "out_of_session_count": stats.out_of_session_count,
        "total_ticks": stats.total_ticks,
        "used_fallback_pref": used_fallback,
    }

    year, month, day = trade_date.split("-")
    output_path = output_dir / year / month / day / f"{stock_id}_{trade_date}.pt"
    save_heatmaps(output_path, heatmaps, metadata)

    return ProcessResult(
        status="ok",
        stock_id=stock_id,
        trade_date=trade_date,
        output_path=str(output_path),
        total_ticks=stats.total_ticks,
        out_of_session_count=stats.out_of_session_count,
        unknown_type_count=stats.unknown_type_count,
        used_fallback_pref=used_fallback,
    )


def load_pref_map_with_auto(
    input_dir: Path, pref_map_path: Path | None, auto_pref: bool
) -> dict[tuple[str, str], float]:
    if pref_map_path:
        pref_map = load_pref_map(pref_map_path)
        logger.info("Loaded pref_map entries: %d", len(pref_map))
        return pref_map

    if not auto_pref:
        logger.info("No pref_map provided; Pref-missing files will be skipped")
        return {}

    candidates = find_pref_candidates(input_dir)
    if not candidates:
        logger.warning("Auto-pref enabled but no candidate files found")
        return {}

    logger.info("Auto-pref using candidate: %s", candidates[0])
    return load_pref_map(candidates[0])


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    pref_map_path: Path | None,
    config: HeatmapConfig,
    workers: int,
    auto_pref: bool,
    show_progress: bool = True,
) -> list[ProcessResult]:
    tick_files = scan_tick_files(input_dir)
    logger.info("Found %d tick files", len(tick_files))

    pref_map = load_pref_map_with_auto(input_dir, pref_map_path, auto_pref)

    results: list[ProcessResult] = []
    processed = 0
    skipped = 0
    errors = 0
    total_unknown = 0
    total_out_of_session = 0
    progress = ProgressBar(len(tick_files), enabled=show_progress)
    progress.update(0)

    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import itertools

        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(
                process_file,
                tick_files,
                itertools.repeat(output_dir),
                itertools.repeat(pref_map),
                itertools.repeat(config),
                chunksize=20,
            )
            for result in iterator:
                results.append(result)
                progress.update()
                if result.status == "ok":
                    processed += 1
                    total_unknown += result.unknown_type_count
                    total_out_of_session += result.out_of_session_count
                elif result.status == "skipped":
                    skipped += 1
                else:
                    errors += 1
    else:
        for path in tick_files:
            result = process_file(path, output_dir, pref_map, config)
            results.append(result)
            progress.update()
            if result.status == "ok":
                processed += 1
                total_unknown += result.unknown_type_count
                total_out_of_session += result.out_of_session_count
            elif result.status == "skipped":
                skipped += 1
            else:
                errors += 1

    progress.finish()
    logger.info(
        "Done. processed=%d skipped=%d errors=%d unknown=%d out_of_session=%d",
        processed,
        skipped,
        errors,
        total_unknown,
        total_out_of_session,
    )
    return results
