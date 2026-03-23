"""Base interfaces for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path


class DataSource(ABC):
    """Abstract interface for market-data downloaders."""

    @abstractmethod
    def download(
        self,
        symbol: str,
        interval: str,
        destination: Path,
        families: Iterable[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Path]:
        """Download raw archives and return written paths."""
