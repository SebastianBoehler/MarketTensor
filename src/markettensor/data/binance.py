"""Binance USD-M futures archive access."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from markettensor.data.base import DataSource
from markettensor.utils.logging import get_logger

LOGGER = get_logger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um"


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _daily_dates(start_date: date, end_date: date) -> list[str]:
    days = []
    current = start_date
    while current <= end_date:
        days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


def _monthly_dates(start_date: date, end_date: date) -> list[str]:
    months: list[str] = []
    year = start_date.year
    month = start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


@dataclass
class BinanceFamilySpec:
    """Path spec for a Binance archive family."""

    family: str
    cadence: str


FAMILY_SPECS = {
    "klines": BinanceFamilySpec(family="klines", cadence="daily"),
    "fundingRate": BinanceFamilySpec(family="fundingRate", cadence="monthly"),
    "metrics": BinanceFamilySpec(family="metrics", cadence="daily"),
}


class BinanceUMFuturesDataSource(DataSource):
    """Downloader for Binance USD-M futures public archives."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def download(
        self,
        symbol: str,
        interval: str,
        destination: Path,
        families: Iterable[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Path]:
        start = _parse_date(start_date or date.today().replace(day=1).isoformat())
        end = _parse_date(end_date or date.today().isoformat())
        written: list[Path] = []
        for family in families:
            spec = FAMILY_SPECS[family]
            tokens = (
                _daily_dates(start, end) if spec.cadence == "daily" else _monthly_dates(start, end)
            )
            for token in tokens:
                file_path = self._download_one(symbol, interval, destination, spec, token)
                if file_path is not None:
                    written.append(file_path)
        return written

    def _download_one(
        self,
        symbol: str,
        interval: str,
        destination: Path,
        spec: BinanceFamilySpec,
        token: str,
    ) -> Path | None:
        file_name = self._build_filename(symbol, interval, spec.family, token)
        family_dir = destination / spec.family / symbol
        family_dir.mkdir(parents=True, exist_ok=True)
        file_path = family_dir / file_name
        if file_path.exists():
            return file_path

        url = self._build_url(symbol, interval, spec.family, spec.cadence, token)
        response = self.session.get(url, timeout=30)
        if response.status_code == 404:
            LOGGER.warning("Archive missing: %s", url)
            return None
        response.raise_for_status()
        file_path.write_bytes(response.content)
        LOGGER.info("Downloaded %s", file_path)
        return file_path

    @staticmethod
    def _build_filename(symbol: str, interval: str, family: str, token: str) -> str:
        if family == "klines":
            return f"{symbol}-{interval}-{token}.zip"
        return f"{symbol}-{family}-{token}.zip"

    @staticmethod
    def _build_url(symbol: str, interval: str, family: str, cadence: str, token: str) -> str:
        if family == "klines":
            return (
                f"{BASE_URL}/{cadence}/{family}/{symbol}/{interval}/{symbol}-{interval}-{token}.zip"
            )
        return f"{BASE_URL}/{cadence}/{family}/{symbol}/{symbol}-{family}-{token}.zip"
