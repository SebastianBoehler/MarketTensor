from __future__ import annotations

from markettensor.labels.direction import make_direction_labels
from markettensor.labels.horizon import make_threshold_labels


def test_direction_labels_are_binary(market_frame):
    labels = make_direction_labels(market_frame, horizon=1)
    assert set(labels.dropna().unique()).issubset({0, 1})


def test_threshold_labels_respect_horizon(market_frame):
    labels = make_threshold_labels(market_frame, horizon=2, threshold=0.0)
    assert len(labels) == len(market_frame)
