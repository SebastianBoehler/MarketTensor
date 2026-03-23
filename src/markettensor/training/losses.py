"""Loss helpers."""

from __future__ import annotations

from torch import nn


def binary_classification_loss() -> nn.Module:
    """Return the default loss for binary classification."""

    return nn.BCEWithLogitsLoss()
