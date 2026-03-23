"""Classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None
) -> dict[str, Any]:
    """Compute core classification metrics."""

    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
    }
    if probabilities is not None and len(np.unique(targets)) > 1:
        from sklearn.metrics import roc_auc_score

        metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
    return metrics
