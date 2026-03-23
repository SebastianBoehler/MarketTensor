"""Training-time preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainOnlyScaler:
    """Train-only feature scaler with serializable state."""

    feature_names: list[str]
    scaler: StandardScaler

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_names: list[str]) -> TrainOnlyScaler:
        scaler = StandardScaler()
        scaler.fit(frame.loc[:, feature_names])
        return cls(feature_names=feature_names, scaler=scaler)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform a frame using fitted training statistics."""

        return self.scaler.transform(frame.loc[:, self.feature_names])

    def state_dict(self) -> dict:
        """Return JSON-serializable scaler metadata."""

        return {
            "feature_names": self.feature_names,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "var": self.scaler.var_.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> TrainOnlyScaler:
        """Restore a scaler from serialized state."""

        scaler = StandardScaler()
        scaler.mean_ = np.asarray(state["mean"])
        scaler.scale_ = np.asarray(state["scale"])
        scaler.var_ = np.asarray(state["var"])
        scaler.n_features_in_ = len(state["feature_names"])
        return cls(feature_names=list(state["feature_names"]), scaler=scaler)
