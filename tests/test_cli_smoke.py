from __future__ import annotations

from pathlib import Path


def test_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    for script_name in [
        "download_data.py",
        "build_dataset.py",
        "train.py",
        "evaluate.py",
        "backtest.py",
        "export_onnx.py",
    ]:
        assert (root / "scripts" / script_name).exists()
