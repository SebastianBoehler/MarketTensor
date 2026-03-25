from __future__ import annotations

import pandas as pd
import pytest

from markettensor.evaluation.benchmark_tables import summarize_grouped_metrics


def test_summarize_grouped_metrics_computes_mean_and_std():
    frame = pd.DataFrame(
        {
            "label": ["CNN", "CNN", "Persistence"],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "accuracy": [0.52, 0.54, 0.5],
            "roc_auc": [0.51, 0.53, 0.5],
            "cumulative_return": [0.2, 0.4, 0.1],
            "sharpe": [0.1, 0.3, 0.0],
            "balanced_accuracy": [0.52, 0.54, 0.5],
            "f1": [0.5, 0.55, 0.48],
            "max_drawdown": [-0.2, -0.1, -0.25],
            "hit_rate": [0.51, 0.52, 0.5],
            "turnover": [1.0, 1.1, 1.0],
        }
    )
    summary = summarize_grouped_metrics(frame, group_columns=["label", "symbol"])
    row = summary[summary["label"] == "CNN"].iloc[0]
    assert row["folds"] == 2
    assert row["accuracy_mean"] == 0.53
    assert row["sharpe_std"] == pytest.approx(0.1)
