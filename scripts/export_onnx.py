"""Export a trained torch model to ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from markettensor.inference.export import export_torch_onnx
from markettensor.models.registry import build_model
from markettensor.training.loop import latest_run_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.runs_dir)
    run_dir = latest_run_path(root) if args.run_id == "latest" else root / args.run_id
    config = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"), resolve=True)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if not (run_dir / "model.pt").exists():
        raise RuntimeError("ONNX export is only available for torch models in v1.")

    input_dim = len(metadata["feature_columns"])
    model = build_model(config["model"], input_dim=input_dim)
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    lookback = int(config["model"]["lookback"])
    export_torch_onnx(model, input_shape=(1, lookback, input_dim), path=run_dir / "model.onnx")
    print(run_dir / "model.onnx")


if __name__ == "__main__":
    main()
