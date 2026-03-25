"""Generate publication-style figures from walk-forward summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import bundles, figsizes

from markettensor.utils.config import ensure_dir

OUTPUT_DIR = Path("docs/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--folds-csv", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def configure_style() -> None:
    """Apply a compact publication theme."""

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


def format_plot_label(label: str) -> str:
    """Compress verbose benchmark labels for figure readability."""

    short = label.replace("1D CNN", "CNN").replace("Open Interest", "OI")
    short = short.replace("(OHLCV + ", "(")
    short = short.replace("OHLCV + ", "")
    short = short.replace("(OHLCV)", "")
    short = " ".join(short.split())
    if " (" in short:
        short = short.replace(" (", "\n(")
    return short


def save_summary_figure(summary: pd.DataFrame, output_dir: Path, prefix: str, title: str) -> None:
    """Save a metric comparison chart with fold-level variability."""

    metric_pairs = [
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("cumulative_return", "Cumulative Return"),
        ("sharpe", "Sharpe"),
    ]
    labels = summary["label"].tolist()
    tick_labels = [format_plot_label(label) for label in labels]
    colors = palette(len(labels))
    figure, axes = plt.subplots(1, len(metric_pairs), figsize=(9.2, 3.1), constrained_layout=True)
    x = np.arange(len(labels))
    for axis, (metric_key, metric_label) in zip(axes, metric_pairs, strict=True):
        means = summary[f"{metric_key}_mean"].to_numpy()
        stds = summary[f"{metric_key}_std"].to_numpy()
        axis.bar(x, means, yerr=stds, capsize=3, color=colors, width=0.65)
        axis.set_xticks(x)
        axis.set_xticklabels(tick_labels, rotation=18, ha="right")
        axis.set_title(metric_label)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    figure.suptitle(title)
    figure.savefig(output_dir / f"{prefix}_summary.png", bbox_inches="tight")
    plt.close(figure)


def save_fold_trace_figure(folds: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    """Save fold-by-fold traces for accuracy and Sharpe."""

    labels = folds["label"].drop_duplicates().tolist()
    display_labels = {label: format_plot_label(label) for label in labels}
    colors = {label: color for label, color in zip(labels, palette(len(labels)), strict=True)}
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), constrained_layout=True)
    for axis, metric_key, metric_label in zip(
        axes,
        ["accuracy", "sharpe"],
        ["Accuracy by Fold", "Sharpe by Fold"],
        strict=True,
    ):
        for label in labels:
            frame = folds[folds["label"] == label].sort_values("fold")
            axis.plot(
                frame["fold"],
                frame[metric_key],
                marker="o",
                linewidth=1.6,
                color=colors[label],
                label=display_labels[label],
            )
        axis.set_title(metric_label)
        axis.set_xlabel("Fold")
        axis.set_xticks(sorted(folds["fold"].unique()))
    axes[0].set_ylabel("Metric Value")
    axes[1].legend(frameon=False)
    figure.savefig(output_dir / f"{prefix}_fold_traces.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = ensure_dir(OUTPUT_DIR)
    summary = pd.read_csv(args.summary_csv)
    folds = pd.read_csv(args.folds_csv)
    save_summary_figure(summary, output_dir=output_dir, prefix=args.prefix, title=args.title)
    save_fold_trace_figure(folds, output_dir=output_dir, prefix=args.prefix)
    print(output_dir)


if __name__ == "__main__":
    main()
