"""Training metric wrappers."""

from __future__ import annotations

import numpy as np


def binary_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute accuracy from logits."""

    predictions = (logits >= 0).astype(int)
    return float((predictions == targets).mean())
