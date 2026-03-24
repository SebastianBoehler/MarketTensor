"""Run a walk-forward benchmark suite and save result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from markettensor.evaluation.walk_forward_benchmark import (
    WalkForwardArtifacts,
    run_walk_forward_suite,
    summarize_walk_forward_results,
    summary_markdown,
)
from markettensor.pipeline import build_dataset
from markettensor.utils.config import ensure_dir, load_experiment_config, to_container


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--config-name", dest="config_names", action="append", required=True)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--override", dest="overrides", action="append", default=[])
    parser.add_argument("--output-dir", default="docs/results")
    return parser.parse_args()


def run_suite(args: argparse.Namespace) -> WalkForwardArtifacts:
    fold_frames: list[pd.DataFrame] = []
    for config_name in args.config_names:
        overrides = list(args.overrides)
        if args.symbols:
            overrides.append(f"experiment.data.symbols=[{','.join(args.symbols)}]")
        config = to_container(load_experiment_config(config_name, overrides=overrides))
        dataset, feature_columns = build_dataset(config)
        fold_frames.append(
            run_walk_forward_suite(
                config=config,
                dataset=dataset,
                feature_columns=feature_columns,
                suite_name=args.suite_name,
            )
        )

    output_dir = ensure_dir(Path(args.output_dir))
    suite_frame = pd.concat(fold_frames, ignore_index=True)
    summary = summarize_walk_forward_results(suite_frame)
    folds_path = output_dir / f"{args.suite_name}_folds.csv"
    summary_path = output_dir / f"{args.suite_name}_summary.csv"
    markdown_path = output_dir / f"{args.suite_name}_summary.md"
    suite_frame.to_csv(folds_path, index=False)
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(summary_markdown(summary), encoding="utf-8")
    return WalkForwardArtifacts(
        suite_name=args.suite_name,
        output_dir=output_dir,
        folds_path=folds_path,
        summary_path=summary_path,
        markdown_path=markdown_path,
    )


def main() -> None:
    artifacts = run_suite(parse_args())
    print(artifacts.summary_path)


if __name__ == "__main__":
    main()
