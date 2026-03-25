"""Trading metric utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from markettensor.evaluation.cost_model import cost_in_return_space


def build_trade_frame(
    predictions: np.ndarray,
    future_returns: np.ndarray,
    symbols: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    holding_period_bars: int = 1,
    non_overlapping: bool = True,
) -> pd.DataFrame:
    """Create a symbol-aware trade frame for metric computation."""

    prediction_array = np.asarray(predictions).astype(int)
    return_array = np.asarray(future_returns, dtype=np.float64)
    symbol_array = (
        np.asarray(symbols)
        if symbols is not None
        else np.repeat("pooled", len(prediction_array))
    )
    timestamp_array = (
        np.asarray(timestamps)
        if timestamps is not None
        else np.arange(len(prediction_array))
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamp_array,
            "symbol": symbol_array,
            "prediction": prediction_array,
            "future_return": return_array,
        }
    ).sort_values(["symbol", "timestamp"])
    if non_overlapping and holding_period_bars > 1:
        frame = _select_non_overlapping_trades(frame, holding_period_bars=holding_period_bars)
    frame["position"] = np.where(frame["prediction"] == 1, 1.0, -1.0)
    return frame.reset_index(drop=True)


def trading_metrics(
    predictions: np.ndarray,
    future_returns: np.ndarray,
    fee_bps: float,
    slippage_bps: float,
    symbols: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    holding_period_bars: int = 1,
    non_overlapping: bool = True,
) -> dict[str, float]:
    """Compute symbol-aware trading metrics from directional predictions."""

    trades = build_trade_frame(
        predictions=predictions,
        future_returns=future_returns,
        symbols=symbols,
        timestamps=timestamps,
        holding_period_bars=holding_period_bars,
        non_overlapping=non_overlapping,
    )
    if trades.empty:
        return {
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "hit_rate": 0.0,
            "cost_adjusted_return": 0.0,
        }

    trades = _apply_trade_costs(trades, fee_bps=fee_bps, slippage_bps=slippage_bps)
    basket_returns = (
        trades.groupby("timestamp", observed=True)["net_return"].mean().to_numpy(dtype=np.float64)
    )
    cumulative = np.cumprod(1.0 + basket_returns) - 1.0
    drawdown = cumulative - np.maximum.accumulate(cumulative)
    sharpe = (
        0.0
        if basket_returns.std() == 0.0
        else float(basket_returns.mean() / basket_returns.std() * np.sqrt(252))
    )
    return {
        "cumulative_return": float(cumulative[-1]) if len(cumulative) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "turnover": float(trades["turnover"].mean()) if len(trades) else 0.0,
        "hit_rate": float((trades["gross_return"] > 0).mean()) if len(trades) else 0.0,
        "cost_adjusted_return": float(trades["net_return"].mean()) if len(trades) else 0.0,
    }


def _select_non_overlapping_trades(
    frame: pd.DataFrame,
    holding_period_bars: int,
) -> pd.DataFrame:
    """Downsample each symbol stream so multi-bar returns do not overlap."""

    selected = []
    for _, symbol_frame in frame.groupby("symbol", observed=True):
        selected.append(symbol_frame.iloc[::holding_period_bars].copy())
    return pd.concat(selected, ignore_index=True)


def _apply_trade_costs(
    trades: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    """Apply turnover-based costs independently per symbol."""

    enriched = trades.copy()
    turnover_parts = []
    for _, symbol_frame in enriched.groupby("symbol", observed=True):
        positions = symbol_frame["position"].to_numpy(dtype=np.float64)
        turnover = np.abs(np.diff(positions, prepend=0.0))
        turnover_parts.append(pd.Series(turnover, index=symbol_frame.index))
    enriched["turnover"] = pd.concat(turnover_parts).sort_index()
    enriched["gross_return"] = enriched["position"] * enriched["future_return"]
    enriched["net_return"] = enriched["gross_return"] - enriched["turnover"].map(
        lambda value: cost_in_return_space(value, fee_bps, slippage_bps)
    )
    return enriched
