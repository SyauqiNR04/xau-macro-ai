from __future__ import annotations

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xau.snr import build_signals


def _bars(start: str, periods: int, freq: str, seed: int, *, with_spread: bool = False):
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    steps = rng.normal(0, 0.7, size=periods)
    close = 2400 + np.cumsum(steps) + 2.0 * np.sin(np.arange(periods) / 7.0)
    open_ = np.r_[close[0], close[:-1]]
    wiggle = rng.uniform(0.1, 1.0, size=periods)
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    frame = pd.DataFrame({
        "time": times, "open": open_, "high": high, "low": low, "close": close,
    })
    if with_spread:
        frame["spread"] = 20.0
        frame["spread_points"] = 20.0
        frame["spread_price"] = 0.02
    return frame


class IncrementalSnrTests(unittest.TestCase):
    def test_incremental_tail_matches_full_build(self):
        bars = {
            "m5": _bars("2026-01-01", 5000, "5min", 1, with_spread=True),
            "m15": _bars("2026-01-01", 1700, "15min", 2),
            "h1": _bars("2026-01-01", 450, "1h", 3),
            "h4": _bars("2025-12-01", 210, "4h", 4),
        }

        def fake_load_bars(path, **kwargs):
            for timeframe, frame in bars.items():
                if str(path).endswith(f"_{timeframe}.csv"):
                    return frame.copy()
            raise AssertionError(f"Unexpected data path: {path}")

        args = dict(
            raw_dir="unused",
            symbol_file="xauusdm",
            left=3,
            right=3,
            max_age_days=120,
            tolerance_atr=0.25,
            min_pivots=3,
            max_width_atr=0.8,
            min_body_range_ratio=0.55,
            spread_points_multiplier=0.001,
        )
        with patch("xau.snr.load_bars", side_effect=fake_load_bars):
            full = build_signals(**args)
        checkpoint = full["signal_time"].iloc[4000]
        end_time = full["signal_time"].iloc[-1]
        with patch("xau.snr.load_bars", side_effect=fake_load_bars):
            incremental = build_signals(**args, start_time=checkpoint, end_time=end_time)

        tail = full.loc[full["signal_time"] > checkpoint]
        incremental_tail = incremental.loc[
            (incremental["signal_time"] > checkpoint) &
            (incremental["signal_time"] <= end_time)
        ]
        self.assertEqual(len(tail), len(incremental_tail))
        for column in (
            "signal", "zone_score", "zone_kind", "zone_low", "zone_high",
            "zone_mid", "zone_pivots", "zone_timeframes",
        ):
            pd.testing.assert_series_equal(
                tail[column].reset_index(drop=True),
                incremental_tail[column].reset_index(drop=True),
                check_names=False,
                check_dtype=False,
            )
        self.assertEqual(
            full.iloc[-1]["current_zones_json"],
            incremental.iloc[-1]["current_zones_json"],
        )


if __name__ == "__main__":
    unittest.main()
