"""Evaluate saved run predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
