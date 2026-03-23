"""Horizon-aware label utilities."""

from __future__ import annotations

import pandas as pd


def make_threshold_labels(frame: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """Create binary labels using a return threshold."""

    future_close = frame.groupby("symbol", observed=True)["close"].shift(-horizon)
    future_return = (future_close - frame["close"]) / frame["close"]
    return (future_return > threshold).astype("Int64")
