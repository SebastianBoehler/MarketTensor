from __future__ import annotations

import json

import pandas as pd

from markettensor.evaluation.reporting import compute_equity_curve, load_run_summary, model_label


def test_model_label_humanizes_names():
    assert model_label("cnn1d") == "1D CNN"
    assert model_label("logistic") == "Logistic Regression"


def test_load_run_summary_reads_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "artifact_metadata.json").write_text(
        json.dumps({"model_name": "cnn1d"}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")
    summary = load_run_summary(run_dir)
    assert summary.model_name == "cnn1d"
    assert summary.metrics["accuracy"] == 0.5


def test_compute_equity_curve_adds_series():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "future_return": [0.01, -0.01, 0.02],
            "probability": [0.6, 0.4, 0.7],
        }
    )
    result = compute_equity_curve(frame)
    assert "equity_curve" in result.columns
    assert len(result) == 3
