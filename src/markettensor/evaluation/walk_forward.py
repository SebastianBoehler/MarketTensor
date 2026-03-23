"""Walk-forward split generation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardFold:
    """A single walk-forward split."""

    train_index: pd.Index
    test_index: pd.Index


def generate_walk_forward_splits(
    frame: pd.DataFrame,
    n_splits: int,
    train_size: float,
    test_size: float,
    step_size: float,
    expanding: bool = True,
) -> list[WalkForwardFold]:
    """Generate chronological walk-forward folds."""

    sorted_frame = frame.sort_values("timestamp").reset_index()
    total = len(sorted_frame)
    train_window = max(1, int(total * train_size))
    test_window = max(1, int(total * test_size))
    step_window = max(1, int(total * step_size))
    folds: list[WalkForwardFold] = []

    train_start = 0
    train_end = train_window
    for _ in range(n_splits):
        test_end = min(total, train_end + test_window)
        if test_end <= train_end:
            break
        train_index = sorted_frame.iloc[train_start:train_end]["index"]
        test_index = sorted_frame.iloc[train_end:test_end]["index"]
        folds.append(WalkForwardFold(train_index=train_index, test_index=test_index))
        train_start = 0 if expanding else train_start + step_window
        train_end += step_window
        if train_end >= total:
            break
    return folds
