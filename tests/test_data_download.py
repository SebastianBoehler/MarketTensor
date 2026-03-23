from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import pytest

from markettensor.data.binance import FAMILY_SPECS, BinanceUMFuturesDataSource
from markettensor.data.loaders import load_klines


class FakeResponse:
    def __init__(self, *, content: bytes = b"", text: str = "", status_code: int = 200) -> None:
        self.content = content
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses

    def get(self, url: str, timeout: int = 30) -> FakeResponse:
        return self.responses.get(url, FakeResponse(status_code=404))


def test_checksum_parsing():
    payload = "09a24635f090c9b2ab17f4a61aa3c4d262a7ccc88e96ff0199ce66a54a8ab66b  sample.zip\n"
    checksum = BinanceUMFuturesDataSource._parse_checksum(payload)
    assert checksum == "09a24635f090c9b2ab17f4a61aa3c4d262a7ccc88e96ff0199ce66a54a8ab66b"


def test_downloader_writes_checksum_and_archive(tmp_path: Path):
    archive_content = b"example-binary-payload"
    archive_checksum = hashlib.sha256(archive_content).hexdigest()
    base_url = (
        "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1h/"
        "BTCUSDT-1h-2024-01-01.zip"
    )
    session = FakeSession(
        {
            f"{base_url}.CHECKSUM": FakeResponse(
                text=f"{archive_checksum}  BTCUSDT-1h-2024-01-01.zip\n",
            ),
            base_url: FakeResponse(content=archive_content),
        }
    )
    source = BinanceUMFuturesDataSource(session=session)

    written = source.download(
        symbol="BTCUSDT",
        interval="1h",
        destination=tmp_path,
        families=["klines"],
        start_date="2024-01-01",
        end_date="2024-01-01",
    )

    assert len(written) == 1
    archive_path = written[0]
    assert archive_path.read_bytes() == archive_content
    assert archive_path.with_name(f"{archive_path.name}.CHECKSUM").exists()


def test_downloader_raises_on_checksum_mismatch(tmp_path: Path):
    base_url = (
        "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1h/"
        "BTCUSDT-1h-2024-01-01.zip"
    )
    session = FakeSession(
        {
            f"{base_url}.CHECKSUM": FakeResponse(
                text=(
                    "09a24635f090c9b2ab17f4a61aa3c4d262a7ccc88e96ff0199ce66a54a8ab66b  "
                    "BTCUSDT-1h-2024-01-01.zip\n"
                ),
            ),
            base_url: FakeResponse(content=b"wrong-payload"),
        }
    )
    source = BinanceUMFuturesDataSource(session=session)

    with pytest.raises(ValueError, match="Checksum verification failed"):
        source.download(
            symbol="BTCUSDT",
            interval="1h",
            destination=tmp_path,
            families=["klines"],
            start_date="2024-01-01",
            end_date="2024-01-01",
        )


def test_load_klines_skips_header_row(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    target_dir = raw_dir / "klines" / "BTCUSDT"
    target_dir.mkdir(parents=True)
    archive_path = target_dir / "BTCUSDT-1h-2024-01-01.zip"
    csv_payload = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,trade_count,"
        "taker_buy_base_volume,taker_buy_quote_volume,ignore\n"
        "1704067200000,42000,42100,41900,42050,100,1704070799999,0,0,0,0,0\n"
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("BTCUSDT-1h-2024-01-01.csv", csv_payload)

    frame = load_klines("BTCUSDT", "1h", raw_dir)

    assert len(frame) == 1
    assert frame.iloc[0]["close"] == 42050


def test_plan_downloads_prefers_monthly_klines_for_full_months():
    source = BinanceUMFuturesDataSource(session=FakeSession({}))
    plan = source._plan_downloads(
        spec=FAMILY_SPECS["klines"],
        start=date(2024, 1, 1),
        end=date(2024, 2, 29),
    )
    assert plan == [("monthly", "2024-01"), ("monthly", "2024-02")]
