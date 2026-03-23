"""Leakage-safe timestamp alignment."""

from __future__ import annotations

import pandas as pd

from markettensor.data.schema import validate_frame


def _shift_columns(frame: pd.DataFrame, columns: list[str], lag_bars: int) -> pd.DataFrame:
    shifted = frame.copy()
    shifted[columns] = shifted.groupby("symbol", observed=True)[columns].shift(lag_bars)
    return shifted


def align_feature_frames(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    lag_bars: int = 1,
) -> pd.DataFrame:
    """Align lower-frequency signals to bar timestamps using backward joins."""

    validate_frame(ohlcv, ["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    aligned_frames: list[pd.DataFrame] = []

    for symbol, base_frame in ohlcv.groupby("symbol", observed=True):
        current = base_frame.sort_values("timestamp").copy()
        if funding is not None and not funding.empty:
            funding_frame = funding[funding["symbol"] == symbol].sort_values("timestamp")
            current = pd.merge_asof(
                current,
                funding_frame,
                on="timestamp",
                by="symbol",
                direction="backward",
            )
        if metrics is not None and not metrics.empty:
            metric_frame = metrics[metrics["symbol"] == symbol].sort_values("timestamp")
            current = pd.merge_asof(
                current,
                metric_frame,
                on="timestamp",
                by="symbol",
                direction="backward",
            )
        aligned_frames.append(current)

    aligned = pd.concat(aligned_frames, ignore_index=True)
    lag_columns = [
        column
        for column in aligned.columns
        if column not in {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    ]
    if lag_columns:
        aligned = _shift_columns(aligned, lag_columns, lag_bars=lag_bars)
    return aligned.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
