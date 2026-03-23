"""Funding feature engineering."""

from __future__ import annotations

import pandas as pd


def build_funding_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select lagged funding features from an aligned frame."""

    columns = ["timestamp", "symbol", "funding_rate", "funding_interval_hours"]
    return frame.loc[:, [column for column in columns if column in frame.columns]].copy()
