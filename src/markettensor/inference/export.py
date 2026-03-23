"""Artifact and ONNX export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from markettensor.training.preprocess import TrainOnlyScaler
from markettensor.utils.config import save_json


@dataclass
class InferenceArtifact:
    """Serializable metadata for inference parity."""

    model_name: str
    feature_names: list[str]
    label_name: str
    lookback: int
    scaler_state: dict[str, Any]


def save_artifact_metadata(artifact: InferenceArtifact, path: Path) -> None:
    """Persist inference metadata."""

    save_json(
        {
            "model_name": artifact.model_name,
            "feature_names": artifact.feature_names,
            "label_name": artifact.label_name,
            "lookback": artifact.lookback,
            "scaler_state": artifact.scaler_state,
        },
        path,
    )


def export_torch_onnx(model: torch.nn.Module, input_shape: tuple[int, ...], path: Path) -> None:
    """Export a torch model to ONNX."""

    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(model, dummy_input, path, input_names=["inputs"], output_names=["logits"])


def scaler_from_metadata(metadata: dict[str, Any]) -> TrainOnlyScaler:
    """Recreate a train-only scaler from saved metadata."""

    return TrainOnlyScaler.from_state_dict(metadata["scaler_state"])
