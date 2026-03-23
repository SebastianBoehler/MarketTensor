"""Report trading metrics from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from markettensor.evaluation.trading import trading_metrics
from markettensor.training.loop import latest_run_path, load_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.runs_dir)
    run_dir = latest_run_path(root) if args.run_id == "latest" else root / args.run_id
    predictions = load_predictions(run_dir)
    metrics = trading_metrics(
        probabilities=predictions["probability"].to_numpy(),
        future_returns=predictions["future_return"].to_numpy(),
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
