from __future__ import annotations
from pathlib import Path
import pandas as pd

OHLC = ("open", "high", "low", "close")

def load_bars(path, *, require_spread=False, spread_points_multiplier=0.001):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file missing: {path}")
    df = pd.read_csv(path)
    if "time" not in df:
        raise ValueError(f"{path}: required column 'time' is missing")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for col in OHLC:
        if col not in df:
            raise ValueError(f"{path}: required OHLC column '{col}' is missing")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", *OHLC]).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path}: no valid OHLC rows")
    if (df[list(OHLC)] <= 0).any().any():
        raise ValueError(f"{path}: non-positive OHLC values found")
    if (df.high < df.low).any() or (df.high < df[["open", "close"]].max(axis=1)).any() or (df.low > df[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"{path}: invalid OHLC candle geometry")
    if require_spread:
        if "spread" not in df:
            raise ValueError(f"{path}: spread column missing; refusing zero-spread fallback")
        raw = pd.to_numeric(df["spread"], errors="coerce")
        if raw.isna().any():
            raise ValueError(f"{path}: missing/non-numeric spread values found")
        if (raw < 0).any():
            raise ValueError(f"{path}: negative spread values found")
        if not (raw > 0).any():
            raise ValueError(f"{path}: spread is zero for every row; refusing backtest")
        df["spread_points"] = raw.astype(float)
        df["spread_price"] = df["spread_points"] * float(spread_points_multiplier)
    else:
        raw = pd.to_numeric(df["spread"], errors="coerce") if "spread" in df else pd.Series(0.0, index=df.index)
        df["spread_points"] = raw.fillna(0)
        df["spread_price"] = df["spread_points"] * float(spread_points_multiplier)
    return df

def add_indicators(df, prefix):
    out = df.copy()
    for span in (20, 50, 200):
        out[f"{prefix}_ema{span}"] = out["close"].ewm(span=span, adjust=False).mean()
    prev = out["close"].shift(1)
    tr = pd.concat([out.high-out.low, (out.high-prev).abs(), (out.low-prev).abs()], axis=1).max(axis=1)
    out[f"{prefix}_atr14"] = tr.rolling(14, min_periods=14).mean()
    return out

def align_closed_timeframe(base, higher, prefix, duration):
    """Join only higher-timeframe candles whose close time has passed."""
    left = base.sort_values("signal_time").copy()
    right = higher.copy()
    right["available_at"] = right["time"] + duration
    cols = ["available_at", "close", f"{prefix}_ema20", f"{prefix}_ema50", f"{prefix}_ema200", f"{prefix}_atr14"]
    right = right[cols].rename(columns={"close": f"{prefix}_close"})
    out = pd.merge_asof(left, right.sort_values("available_at"), left_on="signal_time",
                        right_on="available_at", direction="backward", allow_exact_matches=True)
    return out.drop(columns=["available_at"])
