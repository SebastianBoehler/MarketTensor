"""Tree-based tabular baseline."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier


def build_hist_gradient_boosting(
    max_depth: int = 6, learning_rate: float = 0.1
) -> HistGradientBoostingClassifier:
    """Create a histogram gradient boosting classifier."""

    return HistGradientBoostingClassifier(max_depth=max_depth, learning_rate=learning_rate)
