"""Trading metric utilities."""

from __future__ import annotations

import numpy as np

from markettensor.evaluation.cost_model import cost_in_return_space


def trading_metrics(
    probabilities: np.ndarray, future_returns: np.ndarray, fee_bps: float, slippage_bps: float
) -> dict[str, float]:
    """Compute simple directional trading metrics."""

    positions = np.where(probabilities >= 0.5, 1.0, -1.0)
    gross_returns = positions * future_returns
    turnover = float(np.abs(np.diff(positions, prepend=0)).mean())
    net_returns = gross_returns - cost_in_return_space(turnover, fee_bps, slippage_bps)
    cumulative = np.cumprod(1.0 + net_returns) - 1.0
    drawdown = cumulative - np.maximum.accumulate(cumulative)
    sharpe = (
        0.0
        if net_returns.std() == 0
        else float(net_returns.mean() / net_returns.std() * np.sqrt(252))
    )
    return {
        "cumulative_return": float(cumulative[-1]) if len(cumulative) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "turnover": turnover,
        "hit_rate": float((gross_returns > 0).mean()) if len(gross_returns) else 0.0,
        "cost_adjusted_return": float(net_returns.mean()) if len(net_returns) else 0.0,
    }
