"""Run-summary and figure-report helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from markettensor.evaluation.cost_model import cost_in_return_space
from markettensor.evaluation.trading import build_trade_frame


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
        "logistic": "LogReg",
        "cnn1d": "1D CNN",
        "lstm": "LSTM",
        "tcn": "TCN",
        "mlp": "MLP",
        "hgbt": "HGBT",
        "majority": "Majority",
        "random": "Random",
        "persistence": "Persistence",
    }
    return labels.get(model_name, model_name)


def feature_set_label(run_id: str) -> str:
    """Extract a human-readable feature-set label from a run identifier."""

    parts = run_id.split("_")
    if parts and parts[-1].endswith("Z") and "T" in parts[-1]:
        parts = parts[:-1]
    suffix = "_".join(parts[1:]) if len(parts) > 1 else run_id
    tokens = suffix.split("_")
    token_labels = {
        "ohlcv": "OHLCV",
        "funding": "Funding",
        "open": "Open",
        "interest": "Interest",
        "liquidation": "Liquidation",
        "combined": "Combined",
        "all": "All",
    }
    words: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "open" and index + 1 < len(tokens) and tokens[index + 1] == "interest":
            words.append("Open Interest")
            index += 2
            continue
        words.append(token_labels.get(token, token.replace("-", " ").title()))
        index += 1
    return " + ".join(words)


def plot_labels(summaries: list[RunSummary]) -> list[str]:
    """Generate disambiguated plot labels for run summaries."""

    base_labels = [model_label(summary.model_name) for summary in summaries]
    if len(set(base_labels)) == len(base_labels):
        return base_labels
    return [
        f"{base} ({feature_set_label(summary.run_id)})"
        for base, summary in zip(base_labels, summaries, strict=True)
    ]


def compute_equity_curve(
    predictions: pd.DataFrame,
    fee_bps: float = 2.0,
    slippage_bps: float = 1.0,
    holding_period_bars: int = 1,
) -> pd.DataFrame:
    """Compute a cost-adjusted cumulative return curve from saved predictions."""

    ordered = predictions.sort_values(["timestamp", "symbol"]).reset_index(drop=True).copy()
    prediction_source = (
        ordered["prediction"].to_numpy()
        if "prediction" in ordered
        else (ordered["probability"].to_numpy() >= 0.5).astype(int)
    )
    trades = build_trade_frame(
        predictions=prediction_source,
        future_returns=ordered["future_return"].to_numpy(),
        symbols=ordered["symbol"].to_numpy(),
        timestamps=ordered["timestamp"].to_numpy(),
        holding_period_bars=holding_period_bars,
        non_overlapping=True,
    )
    unit_cost = cost_in_return_space(turnover=1.0, fee_bps=fee_bps, slippage_bps=slippage_bps)
    turnover_parts = []
    for _, symbol_frame in trades.groupby("symbol", observed=True):
        positions = symbol_frame["position"].to_numpy(dtype=np.float64)
        turnover_parts.append(
            pd.Series(
                np.abs(np.diff(positions, prepend=0.0)),
                index=symbol_frame.index,
            )
        )
    trades["turnover"] = pd.concat(turnover_parts).sort_index()
    trades["net_return"] = (
        trades["position"] * trades["future_return"] - trades["turnover"] * unit_cost
    )
    basket = trades.groupby("timestamp", observed=True)["net_return"].mean().reset_index()
    basket["equity_curve"] = np.cumprod(1.0 + basket["net_return"].to_numpy()) - 1.0
    return basket
