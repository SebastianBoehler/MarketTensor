from __future__ import annotations

import pytest

from markettensor.evaluation.trading import build_trade_frame, trading_metrics


def test_trading_metrics_use_binary_predictions_for_positions():
    metrics = trading_metrics(
        predictions=[0, 1],
        future_returns=[0.01, 0.01],
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    assert metrics["hit_rate"] == pytest.approx(0.5)
    assert metrics["cumulative_return"] == pytest.approx(-0.0001, abs=1e-4)


def test_build_trade_frame_respects_symbol_boundaries():
    frame = build_trade_frame(
        predictions=[1, 0, 1, 0],
        future_returns=[0.01, 0.01, 0.01, 0.01],
        symbols=["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
        timestamps=[1, 2, 1, 2],
        holding_period_bars=1,
    )
    assert frame["symbol"].tolist() == ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"]


def test_build_trade_frame_can_drop_overlapping_horizon_rows():
    frame = build_trade_frame(
        predictions=[1, 0, 1, 0, 1],
        future_returns=[0.01] * 5,
        symbols=["BTCUSDT"] * 5,
        timestamps=[1, 2, 3, 4, 5],
        holding_period_bars=2,
        non_overlapping=True,
    )
    assert frame["timestamp"].tolist() == [1, 3, 5]
