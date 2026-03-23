"""Build a processed dataset from raw archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from markettensor.pipeline import build_dataset
from markettensor.utils.config import ensure_dir, load_experiment_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="cnn_ohlcv")
    parser.add_argument("--feature-set")
    parser.add_argument("--symbols", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: list[str] = []
    if args.feature_set:
        overrides.append(f"features={args.feature_set}")
    if args.symbols:
        symbols = ",".join(args.symbols)
        overrides.append(f"experiment.data.symbols=[{symbols}]")
    cfg = load_experiment_config(args.config_name, overrides=overrides)
    config = {
        "data": dict(cfg.data),
        "features": dict(cfg.features),
        "labels": dict(cfg.labels),
        "model": dict(cfg.model),
        "train": dict(cfg.train),
        "eval": dict(cfg.eval),
        "run_name": cfg.run_name,
    }
    dataset, feature_columns = build_dataset(config)
    output_dir = ensure_dir(Path(config["data"]["processed_dir"]))
    dataset_path = output_dir / f"{config['run_name']}_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    save_json(
        {"feature_columns": feature_columns}, output_dir / f"{config['run_name']}_manifest.json"
    )
    print(dataset_path)


if __name__ == "__main__":
    main()
