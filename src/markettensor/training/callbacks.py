"""Training callbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossTracker:
    """Small callback container for loss history."""

    losses: list[float]

    def log(self, value: float) -> None:
        self.losses.append(value)
