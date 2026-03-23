"""Generate academic-style benchmark figures from saved run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
from tueplots import bundles, figsizes

from markettensor.evaluation.reporting import (
    compute_equity_curve,
    load_predictions,
    load_run_summary,
    plot_labels,
)
from markettensor.utils.config import ensure_dir

OUTPUT_DIR = Path("docs/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", dest="run_ids", action="append", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--prefix", default="q1_2024")
    parser.add_argument("--title", default="Pooled Q1 2024 Benchmark Comparison")
    parser.add_argument("--equity-top-k", type=int, default=3)
    return parser.parse_args()


def configure_style() -> None:
    """Apply a publication-style theme."""

    plt.rcParams.update(bundles.neurips2021())
    plt.rcParams.update(figsizes.neurips2021(nrows=1, ncols=2))
    plt.rcParams.update(
        {
            "text.usetex": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 180,
        }
    )


def palette(size: int) -> list[tuple[float, float, float, float]]:
    """Return a deterministic color palette."""

    cmap = plt.get_cmap("tab10")
    return [cmap(index) for index in range(size)]


def save_metric_comparison(
    run_dirs: list[Path],
    output_dir: Path,
    prefix: str,
    title: str,
) -> None:
    """Create a four-panel comparison figure for saved metrics."""

    summaries = [load_run_summary(run_dir) for run_dir in run_dirs]
    labels = plot_labels(summaries)
    metric_pairs = [
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("cumulative_return", "Cumulative Return"),
        ("sharpe", "Sharpe"),
    ]
    colors = palette(len(summaries))
    figure, axes = plt.subplots(1, len(metric_pairs), figsize=(8.8, 3.0))
    for axis, (metric_key, metric_label) in zip(axes, metric_pairs, strict=True):
        values = [summary.metrics.get(metric_key, float("nan")) for summary in summaries]
        x = np.arange(len(labels))
        axis.bar(x, values, color=colors[: len(labels)], width=0.62)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=18, ha="right")
        axis.set_title(metric_label)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_model_comparison.png", bbox_inches="tight")
    plt.close(figure)


def top_run_dirs_by_metric(run_dirs: list[Path], metric_name: str, top_k: int) -> list[Path]:
    """Select the top-k runs by a saved scalar metric."""

    scored = []
    for run_dir in run_dirs:
        summary = load_run_summary(run_dir)
        scored.append((summary.metrics.get(metric_name, float("-inf")), run_dir))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [run_dir for _, run_dir in scored[:top_k]]


def save_equity_curves(
    run_dirs: list[Path],
    output_dir: Path,
    prefix: str,
    top_k: int,
) -> None:
    """Create a cost-adjusted equity curve comparison figure."""

    selected_run_dirs = top_run_dirs_by_metric(run_dirs, metric_name="sharpe", top_k=top_k)
    selected_summaries = [load_run_summary(run_dir) for run_dir in selected_run_dirs]
    labels = plot_labels(selected_summaries)
    colors = palette(len(selected_run_dirs))
    figure, axis = plt.subplots(figsize=(6.3, 3.5))
    for color, run_dir, label in zip(colors, selected_run_dirs, labels, strict=False):
        curve = compute_equity_curve(load_predictions(run_dir))
        grouped = curve.groupby("timestamp", as_index=False)["equity_curve"].mean()
        axis.plot(
            grouped["timestamp"],
            grouped["equity_curve"],
            label=label,
            color=color,
            linewidth=1.8,
        )
    axis.set_title("Average Cost-Adjusted Equity Curve")
    axis.set_ylabel("Cumulative Return")
    axis.set_xlabel("Timestamp")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_equity_curves.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = ensure_dir(OUTPUT_DIR)
    run_dirs = [Path(args.runs_dir) / run_id for run_id in args.run_ids]
    save_metric_comparison(run_dirs, output_dir, prefix=args.prefix, title=args.title)
    save_equity_curves(run_dirs, output_dir, prefix=args.prefix, top_k=args.equity_top_k)
    print(output_dir)


if __name__ == "__main__":
    main()
