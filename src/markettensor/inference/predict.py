"""Inference entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from markettensor.inference.onnx_runtime import run_onnx
from markettensor.training.preprocess import TrainOnlyScaler


def load_metadata(path: Path) -> dict[str, Any]:
    """Load saved artifact metadata."""

    return json.loads(path.read_text(encoding="utf-8"))


def predict_with_torch(
    model: torch.nn.Module,
    feature_frame: pd.DataFrame,
    scaler: TrainOnlyScaler,
    feature_names: list[str],
) -> np.ndarray:
    """Run torch inference from a feature frame."""

    features = scaler.transform(feature_frame)
    tensor = torch.from_numpy(features.astype(np.float32))
    with torch.no_grad():
        logits = model(tensor).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def predict_with_onnx(onnx_path: Path, features: np.ndarray) -> np.ndarray:
    """Run ONNX inference."""

    logits = run_onnx(onnx_path, features)
    return 1.0 / (1.0 + np.exp(-logits))
