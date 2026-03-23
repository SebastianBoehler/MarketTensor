"""Feature-set composition."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from markettensor.features.funding import build_funding_features
from markettensor.features.liquidation import require_liquidation_source
from markettensor.features.ohlcv import build_ohlcv_features
from markettensor.features.open_interest import build_open_interest_features

FEATURE_SET_COMPONENTS = {
    "ohlcv": ["ohlcv"],
    "ohlcv_funding": ["ohlcv", "funding"],
    "ohlcv_open_interest": ["ohlcv", "open_interest"],
    "ohlcv_liquidation": ["ohlcv", "liquidation"],
    "ohlcv_funding_open_interest": ["ohlcv", "funding", "open_interest"],
    "ohlcv_liquidation_open_interest": ["ohlcv", "liquidation", "open_interest"],
    "ohlcv_funding_liquidation": ["ohlcv", "funding", "liquidation"],
    "combined_all": ["ohlcv", "funding", "open_interest", "liquidation"],
}


@dataclass
class FeatureBuildResult:
    """Container for engineered features."""

    frame: pd.DataFrame
    feature_columns: list[str]


def build_feature_set(aligned_frame: pd.DataFrame, feature_config: dict) -> FeatureBuildResult:
    """Compose feature families into a single matrix."""

    families = feature_config["families"]
    parts = [build_ohlcv_features(aligned_frame, lag_bars=feature_config.get("lag_bars", 1))]
    if "funding" in families:
        parts.append(build_funding_features(aligned_frame))
    if "open_interest" in families:
        parts.append(build_open_interest_features(aligned_frame))
    if "liquidation" in families:
        require_liquidation_source()

    merged = parts[0]
    for frame in parts[1:]:
        merge_columns = [
            column for column in frame.columns if column not in {"timestamp", "symbol"}
        ]
        merged = merged.merge(
            frame[["timestamp", "symbol", *merge_columns]], on=["timestamp", "symbol"], how="left"
        )

    feature_columns = [column for column in merged.columns if column not in {"timestamp", "symbol"}]
    return FeatureBuildResult(frame=merged, feature_columns=feature_columns)
