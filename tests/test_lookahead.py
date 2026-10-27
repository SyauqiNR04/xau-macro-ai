import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xau.data import load_bars, align_closed_timeframe
from xau.snr import confirmed_pivots


def test_pivot_not_available_until_right_bars_close():
    t = pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC")
    high = [1, 2, 3, 4, 9, 5, 4, 3, 2, 3, 2, 1]
    low = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 5, 6]
    df = pd.DataFrame({"time": t, "open": low, "high": high, "low": low, "close": low, "m15_atr14": 1.0})
    piv = confirmed_pivots(df, "m15", left=2, right=2)
    p = piv[(piv.kind == "RESISTANCE") & (piv.pivot_time == t[4])].iloc[0]
    assert p.confirmed_at == t[6] + pd.Timedelta(minutes=15)


def test_higher_timeframe_candle_must_be_closed():
    t = pd.Timestamp("2026-01-01 01:15", tz="UTC")
    base = pd.DataFrame({"signal_time": [t]})
    higher = pd.DataFrame({"time": [pd.Timestamp("2026-01-01 01:00", tz="UTC")],
        "close": [100.0], "h1_ema20": [2.0], "h1_ema50": [1.0], "h1_ema200": [0.5], "h1_atr14": [1.0]})
    out = align_closed_timeframe(base, higher, "h1", pd.Timedelta(hours=1))
    assert pd.isna(out.iloc[0]["h1_close"])


def test_missing_spread_fails_hard(tmp_path):
    p = tmp_path / "bars.csv"
    pd.DataFrame({"time": ["2026-01-01T00:00:00Z"], "open": [1], "high": [2], "low": [0.5], "close": [1.5]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="spread column missing"):
        load_bars(p, require_spread=True)


def test_all_zero_spread_fails_hard(tmp_path):
    p = tmp_path / "bars.csv"
    pd.DataFrame({"time": ["2026-01-01T00:00:00Z"], "open": [1], "high": [2], "low": [0.5], "close": [1.5], "spread": [0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="zero for every row"):
        load_bars(p, require_spread=True)
