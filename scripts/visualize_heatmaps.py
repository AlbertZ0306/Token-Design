#!/usr/bin/env python3
"""
Heatmap Visualization Script

Visualizes tensor-formatted heatmap files stored in data/processed/heatmaps/.

Heatmap Structure:
- Shape: (239, 2, 32, 32)
  - 239 time slots (trading hours)
  - 2 channels: 0=Buy (B), 1=Sell (S)
  - 32x32 spatial grid (price x volume)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def load_heatmap(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a .pt file and extract data + metadata.

    Args:
        path: Path to the .pt file

    Returns:
        Tuple of (heatmap_array, metadata_dict)
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("torch is required to load heatmap files")

    data = torch.load(path, weights_only=False)
    heatmaps = data["heatmaps"]
    metadata = data["metadata"]

    # Convert torch tensor to numpy array
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.numpy()

    return heatmaps, metadata


def print_metadata(metadata: dict[str, Any]) -> None:
    """Print metadata summary.

    Args:
        metadata: Dictionary containing heatmap metadata
    """
    print("\n" + "=" * 50)
    print("Heatmap Metadata")
    print("=" * 50)
    for key, value in sorted(metadata.items()):
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")


def plot_grid(
    heatmaps: np.ndarray,
    metadata: dict[str, Any],
    channel: int | str = "both",
    colormap: str = "viridis",
    figsize: tuple[float, float] = (20, 30),
    max_slots: int | None = None,
) -> Figure:
    """Create grid view of all time slots.

    Args:
        heatmaps: Array of shape (t_slots, 2, height, width)
        metadata: Metadata dictionary
        channel: Channel to display (0=B, 1=S, or 'both')
        colormap: Matplotlib colormap name
        figsize: Figure size
        max_slots: Maximum number of slots to display (None for all)

    Returns:
        Matplotlib Figure object
    """
    t_slots, num_channels, height, width = heatmaps.shape

    if max_slots is not None:
        t_slots = min(t_slots, max_slots)

    # Determine number of columns based on channel selection
    if channel == "both":
        num_cols = 2
        channels_to_plot = [0, 1]
        titles = ["Buy (B)", "Sell (S)"]
    elif channel == 0:
        num_cols = 1
        channels_to_plot = [0]
        titles = ["Buy (B)"]
    else:  # channel == 1
        num_cols = 1
        channels_to_plot = [1]
        titles = ["Sell (S)"]

    # Calculate grid dimensions
    num_rows = math.ceil(t_slots / num_cols)
    num_cols_grid = num_cols * 2 if channel == "both" else 2

    fig, axes = plt.subplots(num_rows, num_cols_grid, figsize=figsize)
    fig.suptitle(
        f"Heatmap Grid: {metadata.get('stock_id', 'N/A')} - {metadata.get('trade_date', 'N/A')}",
        fontsize=14,
        fontweight="bold",
    )

    if num_rows == 1 and num_cols_grid == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols_grid == 1:
        axes = axes.reshape(-1, 1)

    for slot_idx in range(t_slots):
        for col_idx, (chan_idx, title) in enumerate(zip(channels_to_plot, titles)):
            ax_idx = slot_idx * 2 + col_idx
            row = ax_idx // num_cols_grid
            col = ax_idx % num_cols_grid

            if row >= num_rows:
                continue

            ax = axes[row, col]
            data = heatmaps[slot_idx, chan_idx]

            im = ax.imshow(data, cmap=colormap, aspect="auto", origin="lower")
            ax.set_title(f"Slot {slot_idx}: {title}", fontsize=8)
            ax.set_xlabel("Price", fontsize=6)
            ax.set_ylabel("Volume", fontsize=6)
            ax.tick_params(labelsize=5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for idx in range(t_slots * 2, num_rows * num_cols_grid):
        row = idx // num_cols_grid
        col = idx % num_cols_grid
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig


def plot_single(
    heatmaps: np.ndarray,
    metadata: dict[str, Any],
    slot: int,
    channel: int | str = "both",
    colormap: str = "viridis",
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """Plot a single time slot.

    Args:
        heatmaps: Array of shape (t_slots, 2, height, width)
        metadata: Metadata dictionary
        slot: Time slot index to display
        channel: Channel to display (0=B, 1=S, or 'both')
        colormap: Matplotlib colormap name
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    t_slots, num_channels, height, width = heatmaps.shape

    if slot < 0 or slot >= t_slots:
        raise ValueError(f"Slot {slot} out of range [0, {t_slots - 1}]")

    if channel == "both":
        num_subplots = 2
        channels_to_plot = [0, 1]
        titles = ["Buy (B)", "Sell (S)"]
    elif channel == 0:
        num_subplots = 1
        channels_to_plot = [0]
        titles = ["Buy (B)"]
    else:  # channel == 1
        num_subplots = 1
        channels_to_plot = [1]
        titles = ["Sell (S)"]

    fig, axes = plt.subplots(1, num_subplots, figsize=figsize)
    if num_subplots == 1:
        axes = [axes]

    fig.suptitle(
        f"Time Slot {slot}: {metadata.get('stock_id', 'N/A')} - {metadata.get('trade_date', 'N/A')}",
        fontsize=14,
        fontweight="bold",
    )

    for ax, chan_idx, title in zip(axes, channels_to_plot, titles):
        data = heatmaps[slot, chan_idx]
        im = ax.imshow(data, cmap=colormap, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Price")
        ax.set_ylabel("Volume")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig


def create_animation(
    heatmaps: np.ndarray,
    metadata: dict[str, Any],
    channel: int = 0,
    colormap: str = "viridis",
    interval: int = 50,
    figsize: tuple[float, float] = (8, 6),
) -> animation.FuncAnimation:
    """Create animated visualization through time slots.

    Args:
        heatmaps: Array of shape (t_slots, 2, height, width)
        metadata: Metadata dictionary
        channel: Channel to animate (0=B, 1=S)
        colormap: Matplotlib colormap name
        interval: Animation interval in milliseconds
        figsize: Figure size

    Returns:
        Matplotlib FuncAnimation object
    """
    t_slots = heatmaps.shape[0]
    channel_name = "Buy (B)" if channel == 0 else "Sell (S)"

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        f"Animation: {metadata.get('stock_id', 'N/A')} - {metadata.get('trade_date', 'N/A')} - {channel_name}",
        fontsize=14,
        fontweight="bold",
    )

    # Initial plot
    data = heatmaps[0, channel]
    im = ax.imshow(data, cmap=colormap, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_xlabel("Price")
    ax.set_ylabel("Volume")
    title_text = ax.set_title(f"Time Slot: 0 / {t_slots - 1}")
    plt.colorbar(im, ax=ax)

    def update(frame):
        data = heatmaps[frame, channel]
        im.set_data(data)
        title_text.set_text(f"Time Slot: {frame} / {t_slots - 1}")
        return [im, title_text]

    ani = animation.FuncAnimation(fig, update, frames=t_slots, interval=interval, blit=True)
    return ani


def plot_difference(
    heatmaps: np.ndarray,
    metadata: dict[str, Any],
    slot: int | None = None,
    colormap: str = "RdBu_r",
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Plot Buy-Sell difference.

    Args:
        heatmaps: Array of shape (t_slots, 2, height, width)
        metadata: Metadata dictionary
        slot: Specific slot to display (None for average over all slots)
        colormap: Matplotlib colormap name
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    t_slots = heatmaps.shape[0]

    if slot is not None:
        if slot < 0 or slot >= t_slots:
            raise ValueError(f"Slot {slot} out of range [0, {t_slots - 1}]")
        diff = heatmaps[slot, 0] - heatmaps[slot, 1]
        title = f"Buy - Sell Difference (Slot {slot})"
    else:
        diff = np.mean(heatmaps[:, 0, :, :] - heatmaps[:, 1, :, :], axis=0)
        title = "Average Buy - Sell Difference (All Slots)"

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        f"{metadata.get('stock_id', 'N/A')} - {metadata.get('trade_date', 'N/A')}",
        fontsize=14,
        fontweight="bold",
    )

    # Use symmetric color range
    vmax = max(abs(diff.min()), abs(diff.max()))
    if vmax > 0:
        im = ax.imshow(diff, cmap=colormap, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(diff, cmap=colormap, aspect="auto", origin="lower")

    ax.set_title(title)
    ax.set_xlabel("Price")
    ax.set_ylabel("Volume")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig


def plot_sum(
    heatmaps: np.ndarray,
    metadata: dict[str, Any],
    slot: int | None = None,
    colormap: str = "viridis",
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Plot total trading activity (Buy + Sell).

    Args:
        heatmaps: Array of shape (t_slots, 2, height, width)
        metadata: Metadata dictionary
        slot: Specific slot to display (None for average over all slots)
        colormap: Matplotlib colormap name
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    t_slots = heatmaps.shape[0]

    if slot is not None:
        if slot < 0 or slot >= t_slots:
            raise ValueError(f"Slot {slot} out of range [0, {t_slots - 1}]")
        total = heatmaps[slot, 0] + heatmaps[slot, 1]
        title = f"Total Activity (Slot {slot})"
    else:
        total = np.mean(heatmaps[:, 0, :, :] + heatmaps[:, 1, :, :], axis=0)
        title = "Average Total Activity (All Slots)"

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        f"{metadata.get('stock_id', 'N/A')} - {metadata.get('trade_date', 'N/A')}",
        fontsize=14,
        fontweight="bold",
    )

    im = ax.imshow(total, cmap=colormap, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Price")
    ax.set_ylabel("Volume")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig


def process_single_file(
    input_path: Path,
    mode: str,
    output_dir: Path | None = None,
    **kwargs,
) -> None:
    """Process a single heatmap file.

    Args:
        input_path: Path to input .pt file
        mode: Visualization mode
        output_dir: Output directory for saving figures
        **kwargs: Additional arguments for visualization
    """
    heatmaps, metadata = load_heatmap(input_path)
    t_slots, num_channels, height, width = heatmaps.shape

    print(f"Loaded heatmap from: {input_path}")
    print(f"  Shape: {heatmaps.shape} (t_slots={t_slots}, channels={num_channels}, size={height}x{width})")
    print_metadata(metadata)

    # Extract visualization parameters
    colormap = kwargs.get("colormap", "viridis")
    figsize = kwargs.get("figsize", (12, 8))
    dpi = kwargs.get("dpi", 100)
    slot = kwargs.get("slot", 0)
    channel = kwargs.get("channel", "both")
    max_slots = kwargs.get("max_slots", None)
    interval = kwargs.get("interval", 50)

    fig = None
    ani = None

    if mode == "grid":
        fig = plot_grid(heatmaps, metadata, channel=channel, colormap=colormap, figsize=figsize, max_slots=max_slots)
    elif mode == "single":
        fig = plot_single(heatmaps, metadata, slot=slot, channel=channel, colormap=colormap, figsize=figsize)
    elif mode == "animation":
        channel_for_anim = 0 if channel == "both" else int(channel)
        ani = create_animation(heatmaps, metadata, channel=channel_for_anim, colormap=colormap, interval=interval, figsize=figsize)
    elif mode == "difference":
        fig = plot_difference(heatmaps, metadata, slot=kwargs.get("slot", None), colormap=colormap, figsize=figsize)
    elif mode == "sum":
        fig = plot_sum(heatmaps, metadata, slot=kwargs.get("slot", None), colormap=colormap, figsize=figsize)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Save or show
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{metadata.get('stock_id', 'heatmap')}_{metadata.get('trade_date', 'date')}_{mode}"
        output_path = output_dir / f"{output_name}.png" if fig else output_dir / f"{output_name}.gif"
        print(f"Saving to: {output_path}")

        if fig:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        elif ani:
            ani.save(output_path, writer="pillow", fps=1000 // interval)
        print("Done.")
    else:
        if fig:
            plt.show()
        elif ani:
            plt.show()


def process_batch(
    input_paths: list[Path],
    mode: str,
    output_dir: Path | None = None,
    **kwargs,
) -> None:
    """Process multiple heatmap files.

    Args:
        input_paths: List of paths to input .pt files
        mode: Visualization mode
        output_dir: Output directory for saving figures
        **kwargs: Additional arguments for visualization
    """
    for i, input_path in enumerate(input_paths, 1):
        print(f"\n[{i}/{len(input_paths)}] Processing: {input_path.name}")
        try:
            process_single_file(input_path, mode, output_dir, **kwargs)
        except Exception as e:
            logger.error(f"Failed to process {input_path}: {e}")
            continue


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize tensor-formatted heatmap files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grid view of all time slots
  python -m scripts.visualize_heatmaps --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt --mode grid

  # Single time slot
  python -m scripts.visualize_heatmaps --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt --mode single --slot 120

  # Animation
  python -m scripts.visualize_heatmaps --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt --mode animation --interval 50

  # Buy-Sell difference
  python -m scripts.visualize_heatmaps --input data/processed/heatmaps/2021/01/12/000088_2021-01-12.pt --mode difference

  # Batch process with output
  python -m scripts.visualize_heatmaps --input "data/processed/heatmaps/2021/01/12/*.pt" --mode grid --output outputs/visualizations
        """,
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file path (supports wildcards for batch processing)",
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["grid", "single", "animation", "difference", "sum"],
        default="grid",
        help="Visualization mode (default: grid)",
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory for saving figures (if not specified, display interactively)",
    )

    parser.add_argument(
        "--slot",
        type=int,
        default=0,
        help="Time slot index for 'single' mode (default: 0)",
    )

    parser.add_argument(
        "--channel",
        default="both",
        help="Channel to display (0=Buy, 1=Sell, both) (default: both)",
    )

    parser.add_argument(
        "--colormap",
        default="viridis",
        help="Matplotlib colormap name (default: viridis)",
    )

    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 8],
        help="Figure size width height (default: 12 8)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Output resolution in DPI (default: 100)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Animation interval in milliseconds (default: 50)",
    )

    parser.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help="Maximum number of slots to display in grid mode (default: all)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve input paths
    input_str = args.input
    input_path = Path(input_str)

    if "*" in input_str:
        # Handle wildcards
        parent_dir = input_path.parent
        pattern = input_path.name
        input_paths = list(parent_dir.glob(pattern))
        if not input_paths:
            print(f"No files found matching pattern: {input_str}")
            return 1
    elif input_path.is_dir():
        # Process all .pt files in directory
        input_paths = list(input_path.glob("**/*.pt"))
        if not input_paths:
            print(f"No .pt files found in directory: {input_path}")
            return 1
    else:
        input_paths = [input_path]

    # Prepare kwargs
    kwargs = {
        "colormap": args.colormap,
        "figsize": tuple(args.figsize),
        "dpi": args.dpi,
        "slot": args.slot,
        "channel": args.channel,
        "interval": args.interval,
        "max_slots": args.max_slots,
    }

    # Validate channel argument
    if args.channel not in ("0", "1", "both"):
        print(f"Invalid channel: {args.channel}. Must be 0, 1, or 'both'")
        return 1

    # Process files
    if len(input_paths) > 1:
        print(f"Found {len(input_paths)} files to process")
        process_batch(input_paths, args.mode, Path(args.output) if args.output else None, **kwargs)
    else:
        process_single_file(input_paths[0], args.mode, Path(args.output) if args.output else None, **kwargs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
