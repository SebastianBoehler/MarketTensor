"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def market_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    rows = []
    for symbol, offset in [("BTCUSDT", 0.0), ("ETHUSDT", 10.0)]:
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": 100 + offset + index,
                    "high": 101 + offset + index,
                    "low": 99 + offset + index,
                    "close": 100.5 + offset + index,
                    "volume": 1000 + index,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def aligned_frame(market_frame: pd.DataFrame) -> pd.DataFrame:
    frame = market_frame.copy()
    frame["funding_rate"] = np.linspace(-0.001, 0.001, len(frame))
    frame["funding_interval_hours"] = 8
    frame["open_interest"] = np.linspace(1000, 2000, len(frame))
    frame["open_interest_value"] = np.linspace(1_000_000, 2_000_000, len(frame))
    return frame
