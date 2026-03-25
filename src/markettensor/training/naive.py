"""Deterministic naive baselines for directional forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NaivePredictionResult:
    """Predicted probabilities and binary decisions for a naive baseline."""

    probability: np.ndarray
    prediction: np.ndarray


def majority_baseline(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> NaivePredictionResult:
    """Predict the train-set majority class for every test row."""

    positive_rate = float(train_frame["target"].mean())
    label = int(positive_rate >= 0.5)
    probability = np.full(len(test_frame), positive_rate, dtype=np.float64)
    prediction = np.full(len(test_frame), label, dtype=np.int64)
    return NaivePredictionResult(probability=probability, prediction=prediction)


def random_baseline(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    seed: int,
) -> NaivePredictionResult:
    """Sample deterministic Bernoulli predictions using the train-set class balance."""

    positive_rate = float(train_frame["target"].mean())
    rng = np.random.default_rng(seed)
    probability = np.full(len(test_frame), positive_rate, dtype=np.float64)
    prediction = rng.binomial(1, positive_rate, size=len(test_frame)).astype(np.int64)
    return NaivePredictionResult(probability=probability, prediction=prediction)


def persistence_baseline(
    context_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> NaivePredictionResult:
    """Predict that the future move follows the most recent realized return sign."""

    ordered = context_frame.sort_values(["symbol", "timestamp"]).copy()
    realized_return = ordered.groupby("symbol", observed=True)["close"].pct_change()
    signal = realized_return.ge(0.0).astype(np.float64)
    signal = signal.groupby(ordered["symbol"], observed=True).transform(
        lambda values: values.ffill()
    )
    ordered["probability"] = signal.fillna(0.5)
    lookup = ordered.loc[:, ["timestamp", "symbol", "probability"]]
    merged = test_frame.loc[:, ["timestamp", "symbol"]].merge(
        lookup,
        on=["timestamp", "symbol"],
        how="left",
    )
    probability = merged["probability"].fillna(0.5).to_numpy(dtype=np.float64)
    prediction = (probability >= 0.5).astype(np.int64)
    return NaivePredictionResult(probability=probability, prediction=prediction)
