"""OHLCV feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_ohlcv_features(frame: pd.DataFrame, lag_bars: int = 1) -> pd.DataFrame:
    """Build lagged OHLCV-derived features."""

    features = frame.loc[:, ["timestamp", "symbol"]].copy()
    grouped = frame.groupby("symbol", observed=True)
    features["open_lag1"] = grouped["open"].shift(lag_bars)
    features["high_lag1"] = grouped["high"].shift(lag_bars)
    features["low_lag1"] = grouped["low"].shift(lag_bars)
    features["close_lag1"] = grouped["close"].shift(lag_bars)
    features["volume_lag1"] = grouped["volume"].shift(lag_bars)
    features["log_return_1"] = grouped["close"].pct_change().shift(lag_bars)
    features["range_pct"] = (
        ((frame["high"] - frame["low"]) / frame["close"]).groupby(frame["symbol"]).shift(lag_bars)
    )
    features["body_pct"] = (
        ((frame["close"] - frame["open"]) / frame["open"]).groupby(frame["symbol"]).shift(lag_bars)
    )
    features["volume_zscore_24"] = (
        grouped["volume"]
        .transform(
            lambda series: (series - series.rolling(24).mean()) / series.rolling(24).std(ddof=0)
        )
        .shift(lag_bars)
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    return features
