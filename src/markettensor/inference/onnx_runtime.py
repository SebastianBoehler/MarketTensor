"""ONNX inference wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


def run_onnx(path: Path, features: np.ndarray) -> np.ndarray:
    """Run ONNX inference for a batch of features."""

    session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    return session.run(None, {"inputs": features.astype(np.float32)})[0]
