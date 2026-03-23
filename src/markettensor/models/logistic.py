"""Logistic regression baseline."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_logistic_regression(max_iter: int = 500) -> LogisticRegression:
    """Create a logistic regression classifier."""

    return LogisticRegression(max_iter=max_iter, class_weight="balanced")
