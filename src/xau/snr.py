from __future__ import annotations
import numpy as np
import pandas as pd
from .data import add_indicators, align_closed_timeframe, load_bars


def confirmed_pivots(df, tf, left=3, right=3):
    """Vectorized pivot discovery; confirmation is delayed until right bars close."""
    high, low = df["high"], df["low"]
    is_high, is_low = pd.Series(True, index=df.index), pd.Series(True, index=df.index)
    for k in range(1, left + 1):
        is_high &= high > high.shift(k)
        is_low &= low < low.shift(k)
    for k in range(1, right + 1):
        is_high &= high > high.shift(-k)
        is_low &= low < low.shift(-k)
    atr = df[f"{tf}_atr14"]
    valid = atr.notna() & (atr > 0)
    delay = {"m15": pd.Timedelta(minutes=15), "h1": pd.Timedelta(hours=1)}[tf]
    frames = []
    for mask, kind, price_col in ((is_high & valid, "RESISTANCE", "high"),
                                  (is_low & valid, "SUPPORT", "low")):
        idx = np.flatnonzero(mask.to_numpy())
        idx = idx[idx + right < len(df)]
        if not len(idx):
            continue
        confirm_times = df["time"].iloc[idx + right].reset_index(drop=True) + delay
        frames.append(pd.DataFrame({
            "confirmed_at": confirm_times,
            "pivot_time": df["time"].iloc[idx].reset_index(drop=True),
            "price": df[price_col].iloc[idx].to_numpy(dtype=float),
            "kind": kind, "timeframe": tf.upper(),
            "pivot_atr": atr.iloc[idx].to_numpy(dtype=float)
        }))
    if not frames:
        return pd.DataFrame(columns=["confirmed_at", "pivot_time", "price", "kind", "timeframe", "pivot_atr"])
    return pd.concat(frames, ignore_index=True).sort_values("confirmed_at").reset_index(drop=True)


def _zones_at(pivots, now, price, atr, max_age_days, tolerance_atr, min_pivots, max_width_atr):
    eligible = pivots[(pivots["confirmed_at"] <= now) &
                      ((now - pivots["pivot_time"]) <= pd.Timedelta(days=max_age_days))]
    zones = []
    for kind in ("SUPPORT", "RESISTANCE"):
        group = eligible.loc[eligible["kind"] == kind].sort_values("price")
        clusters = []
        for p in group.itertuples(index=False):
            tol = max(atr * tolerance_atr, float(p.pivot_atr) * tolerance_atr, 0.01)
            if clusters and abs(float(p.price) - np.mean([float(x.price) for x in clusters[-1]])) <= tol:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        for cluster in clusters:
            if len(cluster) < min_pivots:
                continue
            prices = np.array([float(p.price) for p in cluster])
            low, high = float(prices.min()), float(prices.max())
            if high - low > max(atr * max_width_atr, 0.01):
                continue
            if low - atr * tolerance_atr <= price <= high + atr * tolerance_atr:
                zones.append({"kind": kind, "low": low, "high": high, "mid": float(prices.mean()),
                              "pivots": len(cluster), "timeframes": len(set(p.timeframe for p in cluster))})
    return zones


def build_signals(raw_dir, symbol_file="xauusdm", *, left=3, right=3, max_age_days=120,
                  tolerance_atr=0.25, min_pivots=3, max_width_atr=0.8,
                  min_body_range_ratio=0.55, spread_points_multiplier=0.001):
    raw_dir = str(raw_dir)
    m5 = add_indicators(load_bars(f"{raw_dir}/{symbol_file}_m5.csv", require_spread=True,
                                  spread_points_multiplier=spread_points_multiplier), "m5")
    m15 = add_indicators(load_bars(f"{raw_dir}/{symbol_file}_m15.csv"), "m15")
    h1 = add_indicators(load_bars(f"{raw_dir}/{symbol_file}_h1.csv"), "h1")
    h4 = add_indicators(load_bars(f"{raw_dir}/{symbol_file}_h4.csv"), "h4")
    m5["signal_time"] = m5["time"] + pd.Timedelta(minutes=5)
    m5["prev_high3"] = m5["high"].shift(1).rolling(3, min_periods=3).max()
    m5["prev_low3"] = m5["low"].shift(1).rolling(3, min_periods=3).min()
    df = align_closed_timeframe(m5, m15, "m15", pd.Timedelta(minutes=15))
    df = align_closed_timeframe(df, h1, "h1", pd.Timedelta(hours=1))
    df = align_closed_timeframe(df, h4, "h4", pd.Timedelta(hours=4))
    pivots = pd.concat([confirmed_pivots(m15, "m15", left, right),
                        confirmed_pivots(h1, "h1", left, right)], ignore_index=True)
    pivots = pivots.sort_values("confirmed_at").reset_index(drop=True)

    # Pivot detection and candle predicates are vectorized. Zone clustering is
    # intentionally causal/stateful so future pivots cannot alter past zones.
    signal = np.zeros(len(df), dtype=np.int8)
    zone_score = np.zeros(len(df), dtype=np.int32)
    zone_kind = np.full(len(df), "", dtype=object)
    cursor, active_parts = 0, []
    for i, row in enumerate(df.itertuples(index=False)):
        now = row.signal_time
        while cursor < len(pivots) and pivots.iloc[cursor]["confirmed_at"] <= now:
            active_parts.append(pivots.iloc[cursor])
            cursor += 1
        if not active_parts:
            continue
        active = pd.DataFrame(active_parts)
        active = active.loc[(now - active["pivot_time"]) <= pd.Timedelta(days=max_age_days)]
        active_parts = [p for p in active_parts if now - p["pivot_time"] <= pd.Timedelta(days=max_age_days)]
        atr = row.m5_atr14
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(row.h4_ema200) or not np.isfinite(row.h1_ema50):
            continue
        zones = _zones_at(active, now, float(row.close), float(atr), max_age_days,
                          tolerance_atr, min_pivots, max_width_atr)
        if not zones:
            continue
        candle_range = max(float(row.high - row.low), 1e-9)
        body = abs(float(row.close - row.open))
        lower_wick = min(float(row.open), float(row.close)) - float(row.low)
        upper_wick = float(row.high) - max(float(row.open), float(row.close))
        bull_momentum = row.close > row.open and body / candle_range >= min_body_range_ratio and row.close > row.prev_high3
        bear_momentum = row.close < row.open and body / candle_range >= min_body_range_ratio and row.close < row.prev_low3
        h4_bull, h4_bear = row.h4_ema50 > row.h4_ema200, row.h4_ema50 < row.h4_ema200
        h1_bull, h1_bear = row.h1_ema20 > row.h1_ema50, row.h1_ema20 < row.h1_ema50
        choices = []
        for z in zones:
            if not (row.low <= z["high"] and row.high >= z["low"]):
                continue
            if z["kind"] == "SUPPORT" and h4_bull and h1_bull:
                reject = lower_wick >= max(body * 1.2, 0.15 * atr) and row.close > z["low"]
                if reject or bull_momentum:
                    choices.append((1, z))
            elif z["kind"] == "RESISTANCE" and h4_bear and h1_bear:
                reject = upper_wick >= max(body * 1.2, 0.15 * atr) and row.close < z["high"]
                if reject or bear_momentum:
                    choices.append((-1, z))
        if choices:
            direction, zone = max(choices, key=lambda item: (item[1]["pivots"], item[1]["timeframes"]))
            signal[i], zone_score[i], zone_kind[i] = direction, zone["pivots"] + zone["timeframes"], zone["kind"]
    df["signal"], df["zone_score"], df["zone_kind"] = signal, zone_score, zone_kind
    return df


def simulate_vectorized(df, *, slippage_usd=0.10, atr_stop_mult=1.5, target_r=2.0, max_hold_bars=48):
    """Vectorized barrier checks across candidates/future bars; one pass resolves overlapping trades."""
    columns = ["entry_time", "exit_time", "direction", "entry", "exit", "stop", "target", "exit_reason", "R", "zone_score", "zone_kind"]
    sigs = np.flatnonzero(df["signal"].to_numpy() != 0)
    sigs = sigs[sigs + 1 < len(df)]
    if not len(sigs):
        return pd.DataFrame(columns=columns)
    dirs = df["signal"].to_numpy(dtype=int)[sigs]
    entry_idx = sigs + 1
    entry_open = df["open"].to_numpy(dtype=float)[entry_idx]
    entry_spread = df["spread_price"].to_numpy(dtype=float)[entry_idx]
    risk = df["m5_atr14"].to_numpy(dtype=float)[sigs] * atr_stop_mult
    valid = np.isfinite(risk) & (risk > 0) & np.isfinite(entry_open) & np.isfinite(entry_spread)
    sigs, entry_idx, dirs, entry_open, entry_spread, risk = (x[valid] for x in (sigs, entry_idx, dirs, entry_open, entry_spread, risk))
    if not len(sigs):
        return pd.DataFrame(columns=columns)
    entry = np.where(dirs == 1, entry_open + entry_spread + slippage_usd, entry_open - slippage_usd)
    stop = np.where(dirs == 1, entry - risk, entry + risk)
    target = np.where(dirs == 1, entry + target_r * risk, entry - target_r * risk)
    idx = entry_idx[:, None] + np.arange(0, max_hold_bars)[None, :]
    in_bounds = idx < len(df)
    safe_idx = np.minimum(idx, len(df) - 1)
    op = df["open"].to_numpy(dtype=float)[safe_idx]
    hi = df["high"].to_numpy(dtype=float)[safe_idx]
    lo = df["low"].to_numpy(dtype=float)[safe_idx]
    cl = df["close"].to_numpy(dtype=float)[safe_idx]
    spr = df["spread_price"].to_numpy(dtype=float)[safe_idx]
    long = dirs[:, None] == 1
    ask_op, ask_hi, ask_lo = op + spr, hi + spr, lo + spr
    stop_hit = np.where(long, (op <= stop[:, None]) | (lo <= stop[:, None]),
                        (ask_op >= stop[:, None]) | (ask_hi >= stop[:, None])) & in_bounds
    target_hit = np.where(long, (op >= target[:, None]) | (hi >= target[:, None]),
                          (ask_op <= target[:, None]) | (ask_lo <= target[:, None])) & in_bounds
    hit = stop_hit | target_hit
    has_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1)
    rows = np.arange(len(sigs))
    stop_first = stop_hit[rows, first]  # conservative: stop wins if both barriers hit in same candle
    exit_idx = safe_idx[rows, first].copy()
    hit_open, hit_spread = op[rows, first], spr[rows, first]
    stop_px = np.where(dirs == 1, np.where(hit_open <= stop, hit_open-slippage_usd, stop-slippage_usd),
                       np.where(hit_open+hit_spread >= stop, hit_open+hit_spread+slippage_usd, stop+slippage_usd))
    target_px = np.where(dirs == 1, target-slippage_usd,
                         np.where(hit_open+hit_spread <= target, hit_open+hit_spread+slippage_usd, target+slippage_usd))
    exit_px = np.where(stop_first, stop_px, target_px)
    reason = np.where(stop_first, "SL", "TP").astype(object)
    last_idx = np.minimum(entry_idx + max_hold_bars - 1, len(df)-1)
    no_hit = ~has_hit
    exit_idx[no_hit] = last_idx[no_hit]
    last_close = df["close"].to_numpy(dtype=float)[last_idx]
    last_spread = df["spread_price"].to_numpy(dtype=float)[last_idx]
    exit_px[no_hit] = np.where(dirs[no_hit] == 1, last_close[no_hit]-slippage_usd,
                               last_close[no_hit]+last_spread[no_hit]+slippage_usd)
    reason[no_hit] = "TIME"
    # Sequential only to ensure one open position at a time.
    accepted, next_free = [], -1
    for j in np.argsort(entry_idx):
        if entry_idx[j] > next_free:
            accepted.append(j)
            next_free = int(exit_idx[j])
    a = np.asarray(accepted, dtype=int)
    pnl = np.where(dirs[a] == 1, exit_px[a]-entry[a], entry[a]-exit_px[a])
    out = pd.DataFrame({
        "entry_time": df["time"].to_numpy()[entry_idx[a]],
        "exit_time": df["time"].to_numpy()[exit_idx[a]],
        "direction": np.where(dirs[a] == 1, "LONG", "SHORT"),
        "entry": entry[a], "exit": exit_px[a], "stop": stop[a], "target": target[a],
        "exit_reason": reason[a], "R": pnl/risk[a],
        "zone_score": df["zone_score"].to_numpy()[sigs[a]],
        "zone_kind": df["zone_kind"].to_numpy()[sigs[a]]
    })
    return out
