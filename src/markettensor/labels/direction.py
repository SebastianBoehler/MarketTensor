"""Directional label builders."""

from __future__ import annotations

import pandas as pd


def make_direction_labels(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Create binary up/down labels based on future close returns."""

    future_close = frame.groupby("symbol", observed=True)["close"].shift(-horizon)
    future_return = (future_close - frame["close"]) / frame["close"]
    return (future_return > 0).astype("Int64")
