"""Plot helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_equity_curve(returns: np.ndarray, path: Path) -> None:
    """Save a cumulative return plot."""

    cumulative = np.cumprod(1.0 + returns) - 1.0
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(cumulative)
    axis.set_title("Equity Curve")
    axis.set_xlabel("Step")
    axis.set_ylabel("Cumulative Return")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
