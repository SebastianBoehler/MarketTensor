"""Canonical data schema definitions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CANONICAL_BASE_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OPTIONAL_SIGNAL_COLUMNS = [
    "funding_rate",
    "funding_interval_hours",
    "open_interest",
    "open_interest_value",
    "liquidation_long",
    "liquidation_short",
]


@dataclass(frozen=True)
class DataFrameSchema:
    """Schema contract for aligned market data."""

    required_columns: tuple[str, ...]


CANONICAL_SCHEMA = DataFrameSchema(required_columns=tuple(CANONICAL_BASE_COLUMNS))


def validate_frame(frame: pd.DataFrame, required: list[str] | tuple[str, ...]) -> None:
    """Validate that required columns exist."""

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if frame["timestamp"].isna().any():
        raise ValueError("Timestamp column contains null values.")
