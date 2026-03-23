"""Transaction cost helpers."""

from __future__ import annotations


def cost_in_return_space(turnover: float, fee_bps: float, slippage_bps: float) -> float:
    """Translate turnover into a return-space cost."""

    total_bps = fee_bps + slippage_bps
    return turnover * total_bps / 10_000.0
