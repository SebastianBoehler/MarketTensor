"""Run-summary and figure-report helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from markettensor.evaluation.cost_model import cost_in_return_space


@dataclass(frozen=True)
class RunSummary:
    """Compact benchmark summary for plotting."""

    run_id: str
    model_name: str
    metrics: dict[str, float]
    predictions_path: Path


def load_run_summary(run_dir: Path) -> RunSummary:
    """Load the core benchmark artifacts for one run."""

    metadata = json.loads((run_dir / "artifact_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return RunSummary(
        run_id=run_dir.name,
        model_name=metadata["model_name"],
        metrics=metrics,
        predictions_path=run_dir / "predictions.csv",
    )


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """Load saved predictions with parsed timestamps."""

    return pd.read_csv(run_dir / "predictions.csv", parse_dates=["timestamp"])


def model_label(model_name: str) -> str:
    """Convert internal model names into plot labels."""

    labels = {
        "logistic": "Logistic Regression",
        "cnn1d": "1D CNN",
        "lstm": "LSTM",
        "tcn": "TCN",
        "mlp": "MLP",
        "hgbt": "HistGradientBoosting",
    }
    return labels.get(model_name, model_name)


def compute_equity_curve(
    predictions: pd.DataFrame,
    fee_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    """Compute a cost-adjusted cumulative return curve from saved predictions."""

    ordered = predictions.sort_values(["timestamp", "symbol"]).reset_index(drop=True).copy()
    positions = np.where(ordered["probability"].to_numpy() >= 0.5, 1.0, -1.0)
    turnover = np.abs(np.diff(positions, prepend=0.0))
    unit_cost = cost_in_return_space(turnover=1.0, fee_bps=fee_bps, slippage_bps=slippage_bps)
    net_returns = positions * ordered["future_return"].to_numpy() - turnover * unit_cost
    ordered["equity_curve"] = np.cumprod(1.0 + net_returns) - 1.0
    return ordered
