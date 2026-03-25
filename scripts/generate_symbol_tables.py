"""Generate per-symbol walk-forward summary tables from one suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from markettensor.evaluation.benchmark_tables import (
    fold_level_symbol_metrics,
    summarize_grouped_metrics,
    to_markdown_table,
)
from markettensor.utils.config import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-csv", required=True)
    parser.add_argument("--suite-run-root", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = pd.read_csv(args.folds_csv)
    detailed = fold_level_symbol_metrics(
        folds=folds,
        run_root=Path(args.suite_run_root),
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    summary = summarize_grouped_metrics(detailed, group_columns=["label", "symbol"])
    output_prefix = Path(args.output_prefix)
    detailed.to_csv(
        output_prefix.with_name(f"{output_prefix.name}_per_symbol_folds.csv"),
        index=False,
    )
    summary.to_csv(
        output_prefix.with_name(f"{output_prefix.name}_per_symbol_summary.csv"),
        index=False,
    )
    output_prefix.with_name(f"{output_prefix.name}_per_symbol_summary.md").write_text(
        to_markdown_table(summary),
        encoding="utf-8",
    )
    save_json(
        {
            "folds_csv": args.folds_csv,
            "suite_run_root": args.suite_run_root,
            "fee_bps": args.fee_bps,
            "slippage_bps": args.slippage_bps,
        },
        output_prefix.with_name(f"{output_prefix.name}_per_symbol_manifest.json"),
    )


if __name__ == "__main__":
    main()
