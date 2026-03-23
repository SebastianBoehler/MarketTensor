"""Walk-forward benchmark orchestration."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from markettensor.evaluation.reporting import feature_set_label, model_label
from markettensor.evaluation.splits import apply_purge
from markettensor.evaluation.walk_forward import generate_walk_forward_splits
from markettensor.training.loop import train_experiment
from markettensor.utils.config import ensure_dir, save_json


@dataclass(frozen=True)
class WalkForwardArtifacts:
    """Saved artifacts for one walk-forward suite."""

    suite_name: str
    output_dir: Path
    folds_path: Path
    summary_path: Path
    markdown_path: Path


def benchmark_label(config: dict[str, Any]) -> str:
    """Build a human-readable benchmark label from config."""

    base_label = model_label(config["model"]["name"])
    feature_label = feature_set_label(config["run_name"])
    return base_label if feature_label == "OHLCV" else f"{base_label} ({feature_label})"


def run_walk_forward_suite(
    config: dict[str, Any],
    dataset: pd.DataFrame,
    feature_columns: list[str],
    suite_name: str,
) -> pd.DataFrame:
    """Run walk-forward folds for one experiment config."""

    walk_forward = config["eval"]["walk_forward"]
    folds = generate_walk_forward_splits(
        dataset,
        n_splits=int(walk_forward["n_splits"]),
        train_size=float(walk_forward["train_size"]),
        test_size=float(walk_forward["test_size"]),
        step_size=float(walk_forward["step_size"]),
        expanding=bool(walk_forward["expanding"]),
    )
    if not folds:
        raise ValueError("Walk-forward configuration produced no folds.")

    records: list[dict[str, Any]] = []
    output_dir = ensure_dir(Path(config["train"]["output_dir"]) / "walk_forward" / suite_name)
    label = benchmark_label(config)
    for fold_number, fold in enumerate(folds, start=1):
        train_index = pd.Index(
            apply_purge(fold.train_index, purge_bars=int(config["eval"]["purge_bars"]))
        )
        test_index = pd.Index(fold.test_index)
        train_count, val_count = infer_train_val_counts(
            train_count=len(train_index),
            train_ratio=float(config["data"]["train_ratio"]),
            val_ratio=float(config["data"]["val_ratio"]),
        )
        if train_count <= 0 or val_count <= 0:
            raise ValueError(
                "Fold does not contain enough observations for train/validation split."
            )
        fold_frame = dataset.loc[train_index.union(test_index)].sort_values(["timestamp", "symbol"])
        fold_total = len(train_index) + len(test_index)
        fold_config = copy.deepcopy(config)
        fold_config["data"]["train_ratio"] = train_count / fold_total
        fold_config["data"]["val_ratio"] = val_count / fold_total
        fold_config["train"]["output_dir"] = str(output_dir)
        fold_config["run_name"] = f"{config['run_name']}_wf_fold{fold_number:02d}"
        result = train_experiment(fold_config, fold_frame, feature_columns)
        record = {
            "suite_name": suite_name,
            "config_name": config["run_name"],
            "label": label,
            "fold": fold_number,
            "run_id": result.run_dir.name,
            "train_rows": len(train_index),
            "test_rows": len(test_index),
            "test_start": result.predictions["timestamp"].min().isoformat(),
            "test_end": result.predictions["timestamp"].max().isoformat(),
        }
        record.update(result.metrics)
        records.append(record)
    frame = (
        pd.DataFrame.from_records(records)
        .sort_values(["label", "fold"])
        .reset_index(drop=True)
    )
    save_json(
        {"feature_columns": feature_columns},
        output_dir / f"{config['run_name']}_manifest.json",
    )
    return frame


def infer_train_val_counts(
    train_count: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int]:
    """Convert base split ratios into integer train/validation counts within a fold."""

    pre_test_ratio = train_ratio + val_ratio
    if pre_test_ratio <= 0:
        raise ValueError("Train and validation ratios must sum to a positive value.")
    val_share = val_ratio / pre_test_ratio
    val_count = max(1, int(round(train_count * val_share)))
    if val_count >= train_count:
        val_count = train_count - 1
    return train_count - val_count, val_count


def summarize_walk_forward_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level metrics into a summary table."""

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "cumulative_return",
        "sharpe",
        "max_drawdown",
        "hit_rate",
        "turnover",
    ]
    grouped = fold_results.groupby(["config_name", "label"], as_index=False)
    rows: list[dict[str, Any]] = []
    for _, group in grouped:
        row = {
            "config_name": group["config_name"].iloc[0],
            "label": group["label"].iloc[0],
            "folds": int(len(group)),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False).reset_index(drop=True)


def summary_markdown(summary: pd.DataFrame) -> str:
    """Render a compact markdown summary table."""

    display = summary.loc[
        :,
        [
            "label",
            "folds",
            "accuracy_mean",
            "roc_auc_mean",
            "cumulative_return_mean",
            "sharpe_mean",
        ],
    ].copy()
    rename_map = {
        "label": "Model",
        "folds": "Folds",
        "accuracy_mean": "Accuracy",
        "roc_auc_mean": "ROC-AUC",
        "cumulative_return_mean": "CumReturn",
        "sharpe_mean": "Sharpe",
    }
    display = display.rename(columns=rename_map)
    for column in ["Accuracy", "ROC-AUC", "CumReturn", "Sharpe"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    columns = display.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, divider, *rows])
