"""Dataset conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def make_tabular_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Convert selected features into a 2D matrix."""

    return frame.loc[:, feature_columns].to_numpy(dtype=np.float32)


@dataclass
class WindowedArrays:
    """Windowed feature tensors and labels."""

    features: np.ndarray
    targets: np.ndarray
    row_indices: np.ndarray


def build_windowed_arrays(
    frame: pd.DataFrame, feature_columns: list[str], lookback: int
) -> WindowedArrays:
    """Create rolling windows without crossing symbol boundaries."""

    windows: list[np.ndarray] = []
    targets: list[float] = []
    row_indices: list[int] = []
    for _, symbol_frame in frame.groupby("symbol", observed=True):
        symbol_frame = symbol_frame.sort_values("timestamp").reset_index()
        values = symbol_frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
        labels = symbol_frame["target"].to_numpy(dtype=np.float32)
        original_indices = symbol_frame["index"].to_numpy(dtype=np.int64)
        for row in range(lookback - 1, len(symbol_frame)):
            window = values[row - lookback + 1 : row + 1]
            if np.isnan(window).any() or np.isnan(labels[row]):
                continue
            windows.append(window)
            targets.append(labels[row])
            row_indices.append(int(original_indices[row]))
    return WindowedArrays(
        features=np.asarray(windows),
        targets=np.asarray(targets),
        row_indices=np.asarray(row_indices, dtype=np.int64),
    )


class SequenceDataset(Dataset):
    """Torch dataset for windowed arrays."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def reshape_features_for_model(features: np.ndarray, model_name: str) -> np.ndarray:
    """Adapt window tensors for models that expect flattened inputs."""

    if model_name == "mlp":
        return features.reshape(features.shape[0], -1)
    return features
