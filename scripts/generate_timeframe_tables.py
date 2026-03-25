"""Combine per-timeframe benchmark summaries into one comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from markettensor.evaluation.benchmark_tables import to_markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        required=True,
        help="Format: suite_name:timeframe:summary_csv",
    )
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def parse_suite_spec(spec: str) -> tuple[str, str, Path]:
    parts = spec.split(":", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"Invalid suite spec: {spec}")
    suite_name, timeframe, summary_csv = parts
    return suite_name, timeframe, Path(summary_csv)


def timeframe_order(value: str) -> tuple[int, str]:
    known = {"15m": 0, "1h": 1, "4h": 2}
    return known.get(value, 99), value


def load_timeframe_summaries(specs: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in specs:
        suite_name, timeframe, summary_csv = parse_suite_spec(spec)
        frame = pd.read_csv(summary_csv)
        frame["suite_name"] = suite_name
        frame["timeframe"] = timeframe
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["_timeframe_order"] = combined["timeframe"].map(timeframe_order)
    combined = combined.sort_values(
        by=["_timeframe_order", "label"],
    ).drop(columns="_timeframe_order")
    return combined.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    combined = load_timeframe_summaries(args.suite)
    output_prefix = Path(args.output_prefix)
    combined.to_csv(output_prefix.with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".md").write_text(
        to_markdown_table(combined),
        encoding="utf-8",
    )
    print(output_prefix.with_suffix(".csv"))


if __name__ == "__main__":
    main()
