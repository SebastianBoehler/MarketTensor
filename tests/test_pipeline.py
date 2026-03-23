from __future__ import annotations

import pandas as pd

from markettensor.pipeline import build_dataset


def test_build_dataset_only_loads_requested_signal_families(monkeypatch):
    calls: list[str] = []

    def fake_klines(symbol: str, interval: str, raw_dir):
        calls.append(f"klines:{symbol}")
        timestamps = pd.date_range("2024-01-01", periods=32, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": [symbol] * len(timestamps),
                "open": [1.0 + index for index in range(len(timestamps))],
                "high": [1.5 + index for index in range(len(timestamps))],
                "low": [0.5 + index for index in range(len(timestamps))],
                "close": [1.1 + index for index in range(len(timestamps))],
                "volume": [10.0 + index for index in range(len(timestamps))],
            }
        )

    def fake_funding(symbol: str, raw_dir):
        calls.append(f"funding:{symbol}")
        raise AssertionError("Funding loader should not be used for OHLCV-only features.")

    def fake_metrics(symbol: str, raw_dir):
        calls.append(f"metrics:{symbol}")
        raise AssertionError("Metrics loader should not be used for OHLCV-only features.")

    monkeypatch.setattr("markettensor.pipeline.load_klines", fake_klines)
    monkeypatch.setattr("markettensor.pipeline.load_funding_rates", fake_funding)
    monkeypatch.setattr("markettensor.pipeline.load_metrics", fake_metrics)

    config = {
        "data": {"raw_dir": "unused", "interval": "1h", "symbols": ["BTCUSDT"]},
        "features": {"families": ["ohlcv"], "lag_bars": 1},
        "labels": {"kind": "direction", "horizon": 1},
    }
    dataset, feature_columns = build_dataset(config)
    assert not dataset.empty
    assert feature_columns
    assert calls == ["klines:BTCUSDT"]
