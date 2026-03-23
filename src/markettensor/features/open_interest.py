"""Open-interest feature engineering."""

from __future__ import annotations

import pandas as pd

OPEN_INTEREST_COLUMNS = [
    "open_interest",
    "open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]


def build_open_interest_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select open-interest-style exchange metric features."""

    columns = ["timestamp", "symbol"] + [
        column for column in OPEN_INTEREST_COLUMNS if column in frame.columns
    ]
    return frame.loc[:, columns].copy()
