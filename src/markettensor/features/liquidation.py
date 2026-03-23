"""Liquidation feature hooks."""

from __future__ import annotations


class LiquidationDataUnavailableError(RuntimeError):
    """Raised when liquidation features are requested without a source."""


def require_liquidation_source() -> None:
    """Raise an explicit error for unsupported liquidation ingestion."""

    raise LiquidationDataUnavailableError(
        "Liquidation features are scaffolded but no reproducible historical source is wired yet."
    )
