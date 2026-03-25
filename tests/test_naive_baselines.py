from __future__ import annotations

import pandas as pd

from markettensor.training.naive import (
    majority_baseline,
    persistence_baseline,
    random_baseline,
)


def test_majority_baseline_uses_train_class_balance():
    train = pd.DataFrame({"target": [1, 1, 0, 1]})
    test = pd.DataFrame({"target": [0, 1]})
    result = majority_baseline(train, test)
    assert result.probability.tolist() == [0.75, 0.75]
    assert result.prediction.tolist() == [1, 1]


def test_random_baseline_is_seeded():
    train = pd.DataFrame({"target": [1, 1, 0, 1]})
    test = pd.DataFrame({"target": [0, 1, 0, 1]})
    left = random_baseline(train, test, seed=7)
    right = random_baseline(train, test, seed=7)
    assert left.prediction.tolist() == right.prediction.tolist()


def test_persistence_baseline_uses_last_realized_return_sign():
    context = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T02:00:00Z",
                ]
            ),
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "close": [100.0, 101.0, 99.0],
        }
    )
    test = context.iloc[[1, 2]].copy()
    result = persistence_baseline(context, test)
    assert result.prediction.tolist() == [1, 0]
