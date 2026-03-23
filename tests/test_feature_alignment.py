from __future__ import annotations

import pandas as pd

from markettensor.data.alignment import align_feature_frames


def test_feature_alignment_uses_backward_join(market_frame):
    funding = pd.DataFrame(
        {
            "timestamp": market_frame["timestamp"].iloc[::4].reset_index(drop=True),
            "symbol": ["BTCUSDT"] * 3 + ["ETHUSDT"] * 3,
            "funding_rate": [0.1] * 6,
            "funding_interval_hours": [8] * 6,
        }
    ).sort_values(["symbol", "timestamp"])
    aligned = align_feature_frames(market_frame, funding=funding, metrics=None, lag_bars=1)
    assert "funding_rate" in aligned.columns
    btc_slice = aligned[aligned["symbol"] == "BTCUSDT"].reset_index(drop=True)
    assert btc_slice.loc[1, "funding_rate"] == 0.1
