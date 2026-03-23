"""Train an experiment."""

from __future__ import annotations

import argparse

from markettensor.pipeline import build_dataset
from markettensor.training.loop import train_experiment
from markettensor.utils.config import load_experiment_config, to_container


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="cnn_ohlcv")
    parser.add_argument("--symbols", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: list[str] = []
    if args.symbols:
        symbols = ",".join(args.symbols)
        overrides.append(f"experiment.data.symbols=[{symbols}]")
    cfg = load_experiment_config(args.config_name, overrides=overrides)
    config = to_container(cfg)
    dataset, feature_columns = build_dataset(config)
    result = train_experiment(config, dataset, feature_columns)
    print(result.run_dir)


if __name__ == "__main__":
    main()
