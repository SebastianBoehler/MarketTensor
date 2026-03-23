from __future__ import annotations

from markettensor.evaluation.splits import chronological_split
from markettensor.evaluation.walk_forward import generate_walk_forward_splits


def test_chronological_split_orders_indices(market_frame):
    split = chronological_split(market_frame, train_ratio=0.6, val_ratio=0.2)
    train_timestamps = market_frame.loc[split.train, "timestamp"]
    val_timestamps = market_frame.loc[split.val, "timestamp"]
    test_timestamps = market_frame.loc[split.test, "timestamp"]
    assert train_timestamps.max() <= val_timestamps.min()
    assert val_timestamps.max() <= test_timestamps.min()


def test_walk_forward_generates_multiple_folds(market_frame):
    folds = generate_walk_forward_splits(
        market_frame,
        n_splits=3,
        train_size=0.5,
        test_size=0.2,
        step_size=0.1,
        expanding=True,
    )
    assert len(folds) >= 2
    assert all(len(fold.train_index) > 0 for fold in folds)
