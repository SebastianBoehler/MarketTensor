"""Chronological split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitIndices:
    """Index slices for a temporal split."""

    train: pd.Index
    val: pd.Index
    test: pd.Index


def chronological_split(frame: pd.DataFrame, train_ratio: float, val_ratio: float) -> SplitIndices:
    """Split a time-ordered frame into train, validation, and test indices."""

    sorted_frame = frame.sort_values("timestamp").reset_index()
    total = len(sorted_frame)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train = sorted_frame.iloc[:train_end]["index"]
    val = sorted_frame.iloc[train_end:val_end]["index"]
    test = sorted_frame.iloc[val_end:]["index"]
    return SplitIndices(train=train, val=val, test=test)


def apply_purge(indices: pd.Index, purge_bars: int) -> pd.Index:
    """Drop the last purge bars from a train index."""

    if purge_bars <= 0:
        return indices
    return indices[:-purge_bars]
