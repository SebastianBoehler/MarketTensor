from __future__ import annotations

import numpy as np
import pandas as pd

from markettensor.training.dataset import build_windowed_arrays, reshape_features_for_model


def test_build_windowed_arrays_tracks_source_rows():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "symbol": ["BTCUSDT"] * 4,
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "target": [0.0, 1.0, 0.0, 1.0],
        },
        index=[10, 11, 12, 13],
    )
    arrays = build_windowed_arrays(frame, feature_columns=["feature_a"], lookback=2)
    assert arrays.features.shape == (3, 2, 1)
    assert arrays.targets.tolist() == [1.0, 0.0, 1.0]
    assert arrays.row_indices.tolist() == [11, 12, 13]


def test_reshape_features_for_mlp_flattens_windows():
    features = np.arange(24, dtype=float).reshape(2, 3, 4)
    reshaped = reshape_features_for_model(features, model_name="mlp")
    assert reshaped.shape == (2, 12)
