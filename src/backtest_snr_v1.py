from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

SYMBOL_FILE = "xauusdm"
POINT = 0.001
SLIPPAGE_POINTS = 20
ATR_STOP_MULT = 1.5
TARGET_R = 2.0
MAX_HOLD_BARS = 48
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
MAX_PIVOT_AGE_DAYS = 120
ZONE_TOLERANCE_ATR = 0.25
MIN_ZONE_TOUCHES = 3
OOS_START = pd.Timestamp("2024-01-01", tz="UTC")


def load_bars(tf):
    path = RAW / f"{SYMBOL_FILE}_{tf}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data tidak ditemukan: {path}")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "spread" not in df:
        df["spread"] = 0
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0).clip(lower=0)
    return (df.dropna(subset=["time", "open", "high", "low", "close"])
              .sort_values("time").drop_duplicates("time").reset_index(drop=True))


def indicators(df, prefix):
    df = df.copy()
    for span in (20, 50, 200):
        df[f"{prefix}_ema{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    prev = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    df[f"{prefix}_atr14"] = tr.rolling(14, min_periods=14).mean()
    return df


def confirmed_pivots(df, tf):
    left, right = PIVOT_LEFT, PIVOT_RIGHT
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    out = []
    delay = {"m15": pd.Timedelta(minutes=15), "h1": pd.Timedelta(hours=1)}[tf]
    for i in range(left, len(df) - right):
        hi = highs[i-left:i+right+1]
        lo = lows[i-left:i+right+1]
        atr = df[f"{tf}_atr14"].iloc[i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        pivot_time = df["time"].iloc[i]
        confirmed_at = df["time"].iloc[i+right] + delay
        if highs[i] == hi.max() and np.sum(hi == highs[i]) == 1:
            out.append((confirmed_at, pivot_time, float(highs[i]), "RESISTANCE", tf.upper(), float(atr)))
        if lows[i] == lo.min() and np.sum(lo == lows[i]) == 1:
            out.append((confirmed_at, pivot_time, float(lows[i]), "SUPPORT", tf.upper(), float(atr)))
    return out


def merge_tf(base, tfdf, prefix, duration):
    tfdf = tfdf.copy()
    tfdf["available_at"] = tfdf["time"] + duration
    cols = ["available_at", "close", f"{prefix}_ema20", f"{prefix}_ema50",
            f"{prefix}_ema200", f"{prefix}_atr14"]
    tfdf = tfdf[cols].rename(columns={c: f"{prefix}_{c}" for c in cols if c != "available_at"})
    return pd.merge_asof(base.sort_values("signal_time"), tfdf.sort_values("available_at"),
                         left_on="signal_time", right_on="available_at",
                         direction="backward").drop(columns="available_at")


def zones_at(pivots, now, price, atr):
    eligible = [p for p in pivots if p[0] <= now and now - p[1] <= pd.Timedelta(days=MAX_PIVOT_AGE_DAYS)]
    result = []
    for kind in ("SUPPORT", "RESISTANCE"):
        group = sorted((p for p in eligible if p[3] == kind), key=lambda p: p[2])
        clusters = []
        tolerance = max(atr * ZONE_TOLERANCE_ATR, 0.01)
        for p in group:
            if clusters and abs(p[2] - np.mean([x[2] for x in clusters[-1]])) <= max(
                tolerance, p[5] * ZONE_TOLERANCE_ATR
            ):
                clusters[-1].append(p)
            else:
                clusters.append([p])
        for cluster in clusters:
            if len(cluster) < MIN_ZONE_TOUCHES:
                continue
            prices = [p[2] for p in cluster]
            low, high = min(prices), max(prices)
            if high - low > max(atr * 0.8, 0.01):
                continue
            if low - atr * 0.25 <= price <= high + atr * 0.25:
                result.append({
                    "kind": kind, "low": low, "high": high,
                    "mid": float(np.mean(prices)), "touches": len(cluster),
                    "timeframes": len({p[4] for p in cluster}),
                })
    return result


def build_signals():
    m5 = indicators(load_bars("m5"), "m5")
    m15 = indicators(load_bars("m15"), "m15")
    h1 = indicators(load_bars("h1"), "h1")
    h4 = indicators(load_bars("h4"), "h4")
    m5["signal_time"] = m5["time"] + pd.Timedelta(minutes=5)
    m5["prev_high3"] = m5["high"].shift(1).rolling(3).max()
    m5["prev_low3"] = m5["low"].shift(1).rolling(3).min()
    df = merge_tf(m5, m15, "m15", pd.Timedelta(minutes=15))
    df = merge_tf(df, h1, "h1", pd.Timedelta(hours=1))
    df = merge_tf(df, h4, "h4", pd.Timedelta(hours=4))

    pivots = confirmed_pivots(m15, "m15") + confirmed_pivots(h1, "h1")
    pivots.sort(key=lambda p: p[0])
    pivot_index = 0
    active = []
    signals = np.zeros(len(df), dtype=int)
    zone_scores = np.zeros(len(df), dtype=int)
    zone_kinds = [""] * len(df)

    for i, row in df.iterrows():
        now = row["signal_time"]
        while pivot_index < len(pivots) and pivots[pivot_index][0] <= now:
            active.append(pivots[pivot_index])
            pivot_index += 1
        active = [p for p in active if now - p[1] <= pd.Timedelta(days=MAX_PIVOT_AGE_DAYS)]
        atr = row["m5_atr14"]
        if not np.isfinite(atr) or atr <= 0 or i < 3:
            continue
        zones = zones_at(active, now, float(row["close"]), float(atr))
        if not zones:
            continue

        h4_bull = row["h4_ema50"] > row["h4_ema200"]
        h4_bear = row["h4_ema50"] < row["h4_ema200"]
        h1_bull = row["h1_ema20"] > row["h1_ema50"]
        h1_bear = row["h1_ema20"] < row["h1_ema50"]
        candle_range = max(float(row["high"] - row["low"]), 1e-9)
        body = abs(float(row["close"] - row["open"]))
        lower_wick = min(float(row["open"]), float(row["close"])) - float(row["low"])
        upper_wick = float(row["high"]) - max(float(row["open"]), float(row["close"]))
        bull_momentum = row["close"] > row["open"] and body / candle_range >= 0.55 and row["close"] > row["prev_high3"]
        bear_momentum = row["close"] < row["open"] and body / candle_range >= 0.55 and row["close"] < row["prev_low3"]

        choices = []
        for z in zones:
            touched = row["low"] <= z["high"] and row["high"] >= z["low"]
            if not touched:
                continue
            if z["kind"] == "SUPPORT" and h4_bull and h1_bull:
                rejection = lower_wick >= max(body * 1.2, 0.15 * atr) and row["close"] > z["low"]
                if rejection or bull_momentum:
                    choices.append((1, z))
            elif z["kind"] == "RESISTANCE" and h4_bear and h1_bear:
                rejection = upper_wick >= max(body * 1.2, 0.15 * atr) and row["close"] < z["high"]
                if rejection or bear_momentum:
                    choices.append((-1, z))
        if choices:
            direction, zone = max(choices, key=lambda x: (x[1]["touches"], x[1]["timeframes"]))
            signals[i] = direction
            zone_scores[i] = zone["touches"] + zone["timeframes"]
            zone_kinds[i] = zone["kind"]

    df["signal"] = signals
    df["zone_score"] = zone_scores
    df["zone_kind"] = zone_kinds
    return df


def simulate(df):
    trades = []
    slip = SLIPPAGE_POINTS * POINT
    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]
        direction = int(row["signal"])
        if direction == 0 or not np.isfinite(row["m5_atr14"]):
            i += 1
            continue
        entry_row = df.iloc[i + 1]
        spread = float(entry_row["spread"]) * POINT
        risk = float(row["m5_atr14"]) * ATR_STOP_MULT
        if not np.isfinite(risk) or risk <= 0:
            i += 1
            continue
        raw_open = float(entry_row["open"])
        entry = raw_open + spread + slip if direction == 1 else raw_open - slip
        stop = entry - risk if direction == 1 else entry + risk
        target = entry + TARGET_R * risk if direction == 1 else entry - TARGET_R * risk
        exit_price, reason, exit_time = None, None, None
        last_i = min(i + 1 + MAX_HOLD_BARS, len(df) - 1)
        for j in range(i + 1, last_i + 1):
            r = df.iloc[j]
            op, hi, lo, cl = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
            spr = float(r["spread"]) * POINT
            if direction == 1:
                if op <= stop:
                    exit_price, reason = op - slip, "SL_GAP"
                elif op >= target:
                    exit_price, reason = target - slip, "TP_GAP"
                elif lo <= stop:
                    exit_price, reason = stop - slip, "SL"
                elif hi >= target:
                    exit_price, reason = target - slip, "TP"
            else:
                ask_op, ask_hi, ask_lo = op + spr, hi + spr, lo + spr
                if ask_op >= stop:
                    exit_price, reason = ask_op + slip, "SL_GAP"
                elif ask_op <= target:
                    exit_price, reason = target + slip, "TP_GAP"
                elif ask_hi >= stop:
                    exit_price, reason = stop + slip, "SL"
                elif ask_lo <= target:
                    exit_price, reason = target + slip, "TP"
            if reason:
                exit_time = r["time"]
                break
        if reason is None:
            r = df.iloc[last_i]
            spr = float(r["spread"]) * POINT
            cl = float(r["close"])
            exit_price = cl - slip if direction == 1 else cl + spr + slip
            reason, exit_time = "TIME", r["time"]
        pnl = exit_price - entry if direction == 1 else entry - exit_price
        trades.append({
            "entry_time": entry_row["time"], "exit_time": exit_time,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": entry, "exit": exit_price, "stop": stop, "target": target,
            "exit_reason": reason, "R": pnl / risk,
            "zone_score": int(row["zone_score"]), "zone_kind": row["zone_kind"],
        })
        i = last_i + 1
    return pd.DataFrame(trades)


def summary(trades, label):
    if trades.empty:
        return {"strategy": label, "trades": 0, "win_rate_pct": np.nan,
                "expectancy_R": np.nan, "profit_factor": np.nan,
                "net_R": 0.0, "max_drawdown_R": np.nan}
    r = trades["R"].astype(float)
    equity = r.cumsum()
    dd = equity - equity.cummax()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return {"strategy": label, "trades": len(trades),
            "win_rate_pct": round((r > 0).mean() * 100, 2),
            "expectancy_R": round(r.mean(), 4),
            "profit_factor": round(gp / gl, 4) if gl else np.inf,
            "net_R": round(r.sum(), 4), "max_drawdown_R": round(dd.min(), 4)}


def main():
    print("Membangun sinyal SnR historis tanpa menggunakan pivot sebelum terkonfirmasi...")
    df = build_signals()
    print("Candle:", len(df), "| Rentang:", df["time"].min(), "sampai", df["time"].max())
    print("Sinyal LONG:", int((df["signal"] == 1).sum()), "| SHORT:", int((df["signal"] == -1).sum()))
    trades = simulate(df)
    trades.to_csv(REPORTS / "snr_v1_trades.csv", index=False)
    results = [summary(trades, "SnR FULL")]
    if not trades.empty:
        oos = trades[trades["entry_time"] >= OOS_START]
    else:
        oos = trades
    results.append(summary(oos, "SnR OOS 2024+"))
    report = pd.DataFrame(results)
    report.to_csv(REPORTS / "snr_v1_summary.csv", index=False)
    print("\n=== HASIL BACKTEST SnR V1 ===")
    print(report.to_string(index=False))
    if not trades.empty:
        trades["year"] = pd.to_datetime(trades["entry_time"], utc=True).dt.year
        yearly = trades.groupby("year")["R"].agg(
            trades="count", expectancy_R="mean", net_R="sum"
        ).round(4)
        print("\n=== HASIL TAHUNAN ===")
        print(yearly.to_string())
    print("\nOutput:")
    print(REPORTS / "snr_v1_summary.csv")
    print(REPORTS / "snr_v1_trades.csv")
    print("Catatan: hasil tergantung kualitas spread historis dan asumsi eksekusi; belum bukti profitabilitas.")


if __name__ == "__main__":
    main()
