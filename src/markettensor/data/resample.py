"""Time-series resampling helpers."""

from __future__ import annotations

import pandas as pd


def resample_ohlcv(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample OHLCV bars per symbol."""

    grouped: list[pd.DataFrame] = []
    for symbol, symbol_frame in frame.groupby("symbol", observed=True):
        resampled = (
            symbol_frame.set_index("timestamp")
            .sort_index()
            .resample(interval)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )
        resampled["symbol"] = symbol
        grouped.append(resampled)
    return pd.concat(grouped, ignore_index=True)
