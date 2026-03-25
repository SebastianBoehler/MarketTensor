"""Generate publication-style figures for timeframe benchmark comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tueplots import bundles, figsizes

from markettensor.utils.config import ensure_dir

OUTPUT_DIR = Path("docs/figures")
TIMEFRAME_ORDER = ["15m", "1h", "4h"]
METRICS = [
    ("accuracy_mean", "Accuracy"),
    ("roc_auc_mean", "ROC-AUC"),
    ("cumulative_return_mean", "Cumulative Return"),
    ("sharpe_mean", "Sharpe"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def configure_style() -> None:
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


def label_key(label: str) -> str:
    short = label.replace("1D CNN", "CNN").replace("Open Interest", "OI")
    short = short.replace("(OHLCV + ", "(").replace("OHLCV + ", "")
    return " ".join(short.split())


def palette(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {label: cmap(index) for index, label in enumerate(labels)}


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = ensure_dir(OUTPUT_DIR)
    frame = pd.read_csv(args.summary_csv)
    frame["timeframe"] = pd.Categorical(
        frame["timeframe"],
        categories=TIMEFRAME_ORDER,
        ordered=True,
    )
    frame = frame.sort_values(["timeframe", "label"]).reset_index(drop=True)
    labels = frame["label"].drop_duplicates().tolist()
    colors = palette(labels)

    figure, axes = plt.subplots(1, len(METRICS), figsize=(9.2, 3.2), constrained_layout=True)
    for axis, (metric_key, metric_label) in zip(axes, METRICS, strict=True):
        for label in labels:
            model_frame = frame[frame["label"] == label].sort_values("timeframe")
            axis.plot(
                model_frame["timeframe"].astype(str),
                model_frame[metric_key],
                marker="o",
                linewidth=1.7,
                color=colors[label],
                label=label_key(label),
            )
        axis.set_title(metric_label)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("Metric Value")
    axes[-1].legend(frameon=False)
    figure.suptitle(args.title)
    figure.savefig(output_dir / f"{args.prefix}.png", bbox_inches="tight")
    plt.close(figure)
    print(output_dir / f"{args.prefix}.png")


if __name__ == "__main__":
    main()
