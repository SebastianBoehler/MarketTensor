"""Generate horizon-comparison tables from multiple walk-forward suites."""

from __future__ import annotations

import argparse
from pathlib import Path

from markettensor.evaluation.benchmark_tables import (
    SuiteReference,
    load_horizon_fold_frames,
    summarize_grouped_metrics,
    to_markdown_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        required=True,
        help="Format: suite_name:horizon:folds_csv",
    )
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def parse_suite(value: str) -> SuiteReference:
    suite_name, horizon, folds_csv = value.split(":", maxsplit=2)
    return SuiteReference(
        suite_name=suite_name,
        horizon=int(horizon),
        folds_csv=Path(folds_csv),
        run_root=Path(""),
    )


def main() -> None:
    args = parse_args()
    references = [parse_suite(value) for value in args.suite]
    folds = load_horizon_fold_frames(references)
    summary = summarize_grouped_metrics(folds, group_columns=["label", "horizon"])
    output_prefix = Path(args.output_prefix)
    folds.to_csv(output_prefix.with_name(f"{output_prefix.name}_folds.csv"), index=False)
    summary.to_csv(output_prefix.with_name(f"{output_prefix.name}_summary.csv"), index=False)
    output_prefix.with_name(f"{output_prefix.name}_summary.md").write_text(
        to_markdown_table(summary),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
