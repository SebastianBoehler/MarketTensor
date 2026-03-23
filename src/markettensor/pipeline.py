"""Dataset assembly pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from markettensor.data.alignment import align_feature_frames
from markettensor.data.loaders import load_funding_rates, load_klines, load_metrics
from markettensor.features.combine import build_feature_set
from markettensor.labels.direction import make_direction_labels
from markettensor.labels.horizon import make_threshold_labels


def build_dataset(config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Build a feature-and-label dataset from raw archives."""

    raw_dir = Path(config["data"]["raw_dir"])
    interval = config["data"]["interval"]
    symbols = config["data"]["symbols"]

    ohlcv = pd.concat(
        [load_klines(symbol, interval, raw_dir) for symbol in symbols], ignore_index=True
    )
    funding = pd.concat(
        [load_funding_rates(symbol, raw_dir) for symbol in symbols], ignore_index=True
    )
    metrics = pd.concat([load_metrics(symbol, raw_dir) for symbol in symbols], ignore_index=True)

    aligned = align_feature_frames(
        ohlcv=ohlcv,
        funding=funding if not funding.empty else None,
        metrics=metrics if not metrics.empty else None,
        lag_bars=config["features"]["lag_bars"],
    )
    feature_result = build_feature_set(aligned, config["features"])
    dataset = aligned[["timestamp", "symbol", "close"]].merge(
        feature_result.frame,
        on=["timestamp", "symbol"],
        how="left",
    )

    label_kind = config["labels"]["kind"]
    horizon = int(config["labels"]["horizon"])
    if label_kind == "direction":
        dataset["target"] = make_direction_labels(aligned, horizon=horizon)
    else:
        dataset["target"] = make_threshold_labels(
            aligned,
            horizon=horizon,
            threshold=float(config["labels"]["threshold"]),
        )
    dataset["future_return"] = (
        aligned.groupby("symbol", observed=True)["close"].shift(-horizon) / aligned["close"] - 1.0
    )
    dataset = dataset.dropna(subset=feature_result.feature_columns + ["target", "future_return"])
    return dataset, feature_result.feature_columns
