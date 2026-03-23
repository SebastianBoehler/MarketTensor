"""Binance USD-M futures archive access."""

from __future__ import annotations

import hashlib
from calendar import monthrange
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
    cadence: str | None = None


FAMILY_SPECS = {
    "klines": BinanceFamilySpec(family="klines"),
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
            for cadence, token in self._plan_downloads(spec, start, end):
                file_path = self._download_one(symbol, interval, destination, spec, cadence, token)
                if file_path is not None:
                    written.append(file_path)
        return written

    def _download_one(
        self,
        symbol: str,
        interval: str,
        destination: Path,
        spec: BinanceFamilySpec,
        cadence: str,
        token: str,
    ) -> Path | None:
        file_name = self._build_filename(symbol, interval, spec.family, token)
        family_dir = destination / spec.family / symbol
        family_dir.mkdir(parents=True, exist_ok=True)
        file_path = family_dir / file_name

        url = self._build_url(symbol, interval, spec.family, cadence, token)
        checksum_url = f"{url}.CHECKSUM"
        checksum_path = file_path.with_name(f"{file_path.name}.CHECKSUM")
        expected_checksum = self._download_checksum(checksum_url, checksum_path)
        if expected_checksum is None:
            LOGGER.warning("Archive missing: %s", url)
            return None

        if file_path.exists() and self._verify_checksum(file_path, expected_checksum):
            LOGGER.info("Verified existing archive %s", file_path)
            return file_path

        response = self.session.get(url, timeout=30)
        if response.status_code == 404:
            LOGGER.warning("Archive missing after checksum download: %s", url)
            return None
        response.raise_for_status()
        file_path.write_bytes(response.content)
        if not self._verify_checksum(file_path, expected_checksum):
            raise ValueError(f"Checksum verification failed for {file_path.name}.")
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

    def _plan_downloads(
        self,
        spec: BinanceFamilySpec,
        start: date,
        end: date,
    ) -> list[tuple[str, str]]:
        if spec.family != "klines":
            cadence = spec.cadence or "daily"
            tokens = _daily_dates(start, end) if cadence == "daily" else _monthly_dates(start, end)
            return [(cadence, token) for token in tokens]

        monthly_tokens: list[tuple[str, str]] = []
        daily_tokens: list[tuple[str, str]] = []
        for month_token in _monthly_dates(start, end):
            month_start = _parse_date(f"{month_token}-01")
            month_end = month_start.replace(day=monthrange(month_start.year, month_start.month)[1])
            if start <= month_start and month_end <= end:
                monthly_tokens.append(("monthly", month_token))
                continue
            overlap_start = max(start, month_start)
            overlap_end = min(end, month_end)
            daily_tokens.extend(
                ("daily", token) for token in _daily_dates(overlap_start, overlap_end)
            )
        return [*monthly_tokens, *daily_tokens]

    def _download_checksum(self, checksum_url: str, checksum_path: Path) -> str | None:
        response = self.session.get(checksum_url, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        checksum_path.write_text(response.text, encoding="utf-8")
        return self._parse_checksum(response.text)

    @staticmethod
    def _parse_checksum(payload: str) -> str:
        checksum = payload.strip().split()[0]
        if len(checksum) != 64:
            raise ValueError(f"Unexpected checksum payload: {payload!r}")
        return checksum

    @staticmethod
    def _verify_checksum(file_path: Path, expected_checksum: str) -> bool:
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        return digest == expected_checksum
