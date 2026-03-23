"""Download Binance futures archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from markettensor.data.binance import BinanceUMFuturesDataSource
from markettensor.utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--raw-dir", default="data/raw/binance_um")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--families", nargs="+", default=["klines", "fundingRate", "metrics"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = BinanceUMFuturesDataSource()
    written = source.download(
        symbol=args.symbol,
        interval=args.interval,
        destination=Path(args.raw_dir),
        families=args.families,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    LOGGER.info("Downloaded %d archives.", len(written))


if __name__ == "__main__":
    main()
