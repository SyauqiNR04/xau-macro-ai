from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from xau.snr import confirmed_pivots
from xau.data import align_closed_timeframe


def main():
    t = pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC")
    high = [1, 2, 3, 4, 9, 5, 4, 3, 2, 3, 2, 1]
    low = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 5, 6]
    bars = pd.DataFrame({"time": t, "open": low, "high": high, "low": low,
                         "close": low, "m15_atr14": 1.0})
    piv = confirmed_pivots(bars, "m15", left=2, right=2)
    p = piv[(piv.kind == "RESISTANCE") & (piv.pivot_time == t[4])].iloc[0]
    assert p.confirmed_at == t[6] + pd.Timedelta(minutes=15)
    base = pd.DataFrame({"signal_time": [t[4] + pd.Timedelta(minutes=15)]})
    higher = pd.DataFrame({"time": [t[4]], "close": [10.0], "h1_ema20": [2.0],
        "h1_ema50": [1.0], "h1_ema200": [0.5], "h1_atr14": [1.0]})
    joined = align_closed_timeframe(base, higher, "h1", pd.Timedelta(hours=1))
    assert pd.isna(joined.iloc[0]["h1_close"])
    print("PASS: pivot confirmation delay and higher-timeframe close-time guards")


if __name__ == "__main__":
    main()
