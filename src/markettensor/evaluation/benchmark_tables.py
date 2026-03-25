"""Extended walk-forward result tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from markettensor.evaluation.classification import classification_metrics
from markettensor.evaluation.trading import trading_metrics

METRIC_COLUMNS = [
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


@dataclass(frozen=True)
class SuiteReference:
    """Metadata needed to reload predictions for one saved walk-forward suite."""

    suite_name: str
    folds_csv: Path
    run_root: Path
    horizon: int | None = None


def fold_level_symbol_metrics(
    folds: pd.DataFrame,
    run_root: Path,
    fee_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    """Compute fold-level metrics separately for each traded symbol."""

    records: list[dict[str, Any]] = []
    for row in folds.itertuples(index=False):
        predictions = pd.read_csv(
            run_root / row.run_id / "predictions.csv",
            parse_dates=["timestamp"],
        )
        config = yaml.safe_load((run_root / row.run_id / "config.yaml").read_text(encoding="utf-8"))
        horizon = int(config["labels"]["horizon"])
        for symbol, frame in predictions.groupby("symbol", observed=True):
            record = {
                "suite_name": row.suite_name,
                "config_name": row.config_name,
                "label": row.label,
                "fold": row.fold,
                "symbol": symbol,
            }
            record.update(
                classification_metrics(
                    targets=frame["target"].to_numpy(),
                    predictions=frame["prediction"].to_numpy(),
                    probabilities=frame["probability"].to_numpy(),
                )
            )
            record.update(
                trading_metrics(
                    predictions=frame["prediction"].to_numpy(),
                    future_returns=frame["future_return"].to_numpy(),
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    symbols=frame["symbol"].to_numpy(),
                    timestamps=frame["timestamp"].to_numpy(),
                    holding_period_bars=horizon,
                    non_overlapping=True,
                )
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_grouped_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Aggregate metric columns into mean/std summary rows."""

    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_columns, as_index=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row["folds"] = int(len(group))
        for metric in METRIC_COLUMNS:
            if metric not in group:
                continue
            values = group[metric]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def load_horizon_fold_frames(references: list[SuiteReference]) -> pd.DataFrame:
    """Load multiple suite fold tables and annotate them with the forecast horizon."""

    frames: list[pd.DataFrame] = []
    for reference in references:
        frame = pd.read_csv(reference.folds_csv)
        if reference.horizon is not None:
            frame["horizon"] = reference.horizon
        frame["suite_name"] = reference.suite_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact markdown table with numeric formatting."""

    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    columns = display.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, divider, *rows])
