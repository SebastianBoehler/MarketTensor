from __future__ import annotations

from markettensor.features.ohlcv import build_ohlcv_features
from markettensor.training.preprocess import TrainOnlyScaler


def test_ohlcv_features_are_lagged(market_frame):
    features = build_ohlcv_features(market_frame, lag_bars=1)
    first_valid = features.dropna(subset=["close_lag1"]).iloc[0]
    original = market_frame[(market_frame["symbol"] == first_valid["symbol"])].iloc[0]
    assert first_valid["close_lag1"] == original["close"]


def test_scaler_is_fit_on_train_only(market_frame):
    train = market_frame.iloc[:8].copy()
    test = market_frame.iloc[8:].copy()
    feature_names = ["open", "close", "volume"]
    scaler = TrainOnlyScaler.fit(train, feature_names)
    transformed_test = scaler.transform(test)
    assert transformed_test.shape == (len(test), len(feature_names))
    assert list(scaler.feature_names) == feature_names
