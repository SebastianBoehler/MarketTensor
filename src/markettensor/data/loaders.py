"""Raw archive loaders and dataset assembly."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from markettensor.data.schema import CANONICAL_SCHEMA, validate_frame

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

METRIC_RENAME_MAP = {
    "create_time": "timestamp",
    "sum_open_interest": "open_interest",
    "sum_open_interest_value": "open_interest_value",
}


def _read_zip_csv(path: Path, names: list[str] | None = None) -> pd.DataFrame:
    header = 0 if names is None else None
    return pd.read_csv(path, compression="zip", names=names, header=header)


def load_klines(symbol: str, interval: str, raw_dir: Path) -> pd.DataFrame:
    """Load Binance kline archives for one symbol."""

    files = sorted((raw_dir / "klines" / symbol).glob(f"{symbol}-{interval}-*.zip"))
    if not files:
        raise FileNotFoundError(f"No kline archives found for {symbol} in {raw_dir}.")

    frame = pd.concat((_read_zip_csv(path, KLINE_COLUMNS) for path in files), ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["symbol"] = symbol
    columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    output = frame.loc[:, columns].copy()
    validate_frame(output, CANONICAL_SCHEMA.required_columns)
    return output.sort_values("timestamp").drop_duplicates(["timestamp", "symbol"])


def load_funding_rates(symbol: str, raw_dir: Path) -> pd.DataFrame:
    """Load funding-rate archives for one symbol."""

    files = sorted((raw_dir / "fundingRate" / symbol).glob(f"{symbol}-fundingRate-*.zip"))
    if not files:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "funding_rate", "funding_interval_hours"]
        )

    frame = pd.concat((_read_zip_csv(path) for path in files), ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["calc_time"], unit="ms", utc=True)
    frame["symbol"] = symbol
    frame = frame.rename(columns={"last_funding_rate": "funding_rate"})
    columns = ["timestamp", "symbol", "funding_rate", "funding_interval_hours"]
    return frame.loc[:, columns].sort_values("timestamp").drop_duplicates(["timestamp", "symbol"])


def load_metrics(symbol: str, raw_dir: Path) -> pd.DataFrame:
    """Load metrics archives for one symbol."""

    files = sorted((raw_dir / "metrics" / symbol).glob(f"{symbol}-metrics-*.zip"))
    if not files:
        return pd.DataFrame()

    frame = pd.concat((_read_zip_csv(path) for path in files), ignore_index=True)
    frame = frame.rename(columns=METRIC_RENAME_MAP)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = symbol
    return frame.sort_values("timestamp").drop_duplicates(["timestamp", "symbol"])
