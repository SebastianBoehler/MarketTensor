from __future__ import annotations

import json

from markettensor.inference.export import InferenceArtifact, save_artifact_metadata
from markettensor.training.preprocess import TrainOnlyScaler


def test_artifact_metadata_roundtrip(tmp_path, market_frame):
    scaler = TrainOnlyScaler.fit(market_frame, ["open", "close", "volume"])
    artifact = InferenceArtifact(
        model_name="cnn1d",
        feature_names=["open", "close", "volume"],
        label_name="next_bar",
        lookback=16,
        scaler_state=scaler.state_dict(),
    )
    output = tmp_path / "artifact.json"
    save_artifact_metadata(artifact, output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["model_name"] == "cnn1d"
    assert saved["lookback"] == 16
