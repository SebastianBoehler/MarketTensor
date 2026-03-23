from __future__ import annotations

import pandas as pd

from markettensor.evaluation.walk_forward_benchmark import (
    infer_train_val_counts,
    summarize_walk_forward_results,
    summary_markdown,
)


def test_infer_train_val_counts_reserves_validation_rows():
    train_count, val_count = infer_train_val_counts(train_count=20, train_ratio=0.7, val_ratio=0.15)
    assert train_count > 0
    assert val_count > 0
    assert train_count + val_count == 20


def test_summarize_walk_forward_results_aggregates_metrics():
    folds = pd.DataFrame(
        {
            "config_name": ["cnn_ohlcv", "cnn_ohlcv", "lstm_ohlcv"],
            "label": ["1D CNN", "1D CNN", "LSTM"],
            "accuracy": [0.55, 0.57, 0.54],
            "balanced_accuracy": [0.54, 0.56, 0.53],
            "f1": [0.55, 0.58, 0.54],
            "roc_auc": [0.52, 0.53, 0.56],
            "cumulative_return": [0.2, 0.3, 0.1],
            "sharpe": [1.0, 1.2, 0.8],
            "max_drawdown": [-0.1, -0.2, -0.15],
            "hit_rate": [0.51, 0.52, 0.5],
            "turnover": [0.9, 1.1, 1.0],
        }
    )
    summary = summarize_walk_forward_results(folds)
    assert summary.loc[0, "label"] == "1D CNN"
    assert summary.loc[0, "accuracy_mean"] == 0.56
    assert "sharpe_std" in summary.columns


def test_summary_markdown_contains_table_headers():
    summary = pd.DataFrame(
        {
            "label": ["1D CNN"],
            "folds": [4],
            "accuracy_mean": [0.55],
            "roc_auc_mean": [0.52],
            "cumulative_return_mean": [0.3],
            "sharpe_mean": [1.1],
        }
    )
    table = summary_markdown(summary)
    assert "Model" in table
    assert "Accuracy" in table
