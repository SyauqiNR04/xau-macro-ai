from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

SYMBOL_FILE = "xauusdm"
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
ZONE_ATR_TOLERANCE = 0.25
ZONE_MAX_WIDTH_ATR = 0.80
MAX_PIVOT_AGE_DAYS = 120
MIN_ZONE_SCORE = 3
SIGNAL_LOOKBACK_BARS = 3


def load_bars(timeframe):
    path = RAW / f"{SYMBOL_FILE}_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {path}")

    df = pd.read_csv(path)
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} tidak memiliki kolom: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )
    return df


def add_indicators(df, prefix):
    df = df.copy()
    for span in [20, 50, 200]:
        df[f"{prefix}_ema{span}"] = df["close"].ewm(
            span=span, adjust=False
        ).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df[f"{prefix}_atr14"] = tr.rolling(14, min_periods=14).mean()
    return df


def find_confirmed_pivots(df, timeframe):
    """Pivot only becomes usable after PIVOT_RIGHT candles have closed."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["time"].to_numpy()
    atr = df[f"{timeframe}_atr14"].to_numpy()
    pivots = []

    for i in range(PIVOT_LEFT, len(df) - PIVOT_RIGHT):
        high_window = highs[i - PIVOT_LEFT : i + PIVOT_RIGHT + 1]
        low_window = lows[i - PIVOT_LEFT : i + PIVOT_RIGHT + 1]

        if highs[i] >= np.max(high_window) and np.sum(high_window == highs[i]) == 1:
            if np.isfinite(atr[i]) and atr[i] > 0:
                pivots.append(
                    {
                        "pivot_time": pd.Timestamp(times[i]),
                        "confirmed_at": pd.Timestamp(times[i + PIVOT_RIGHT]),
                        "price": float(highs[i]),
                        "kind": "RESISTANCE",
                        "source_tf": timeframe.upper(),
                        "atr": float(atr[i]),
                    }
                )

        if lows[i] <= np.min(low_window) and np.sum(low_window == lows[i]) == 1:
            if np.isfinite(atr[i]) and atr[i] > 0:
                pivots.append(
                    {
                        "pivot_time": pd.Timestamp(times[i]),
                        "confirmed_at": pd.Timestamp(times[i + PIVOT_RIGHT]),
                        "price": float(lows[i]),
                        "kind": "SUPPORT",
                        "source_tf": timeframe.upper(),
                        "atr": float(atr[i]),
                    }
                )

    return pivots


def cluster_zones(pivots, now, current_atr):
    """Group nearby confirmed pivots without using future candles."""
    if not pivots:
        return pd.DataFrame()

    max_age = pd.Timedelta(days=MAX_PIVOT_AGE_DAYS)
    eligible = [
        p for p in pivots
        if p["confirmed_at"] <= now and now - p["pivot_time"] <= max_age
    ]
    zones = []

    for kind in ["SUPPORT", "RESISTANCE"]:
        group = sorted(
            [p for p in eligible if p["kind"] == kind],
            key=lambda p: p["price"],
        )
        clusters = []

        for pivot in group:
            tolerance = max(
                ZONE_ATR_TOLERANCE * pivot["atr"],
                ZONE_ATR_TOLERANCE * current_atr if np.isfinite(current_atr) else 0,
            )
            if not clusters:
                clusters.append([pivot])
                continue

            existing = clusters[-1]
            center = float(np.mean([x["price"] for x in existing]))
            width_limit = max(
                ZONE_MAX_WIDTH_ATR * pivot["atr"],
                ZONE_MAX_WIDTH_ATR * current_atr if np.isfinite(current_atr) else 0,
            )
            if abs(pivot["price"] - center) <= tolerance and (
                max([x["price"] for x in existing] + [pivot["price"]])
                - min([x["price"] for x in existing] + [pivot["price"]])
            ) <= width_limit:
                existing.append(pivot)
            else:
                clusters.append([pivot])

        for cluster in clusters:
            prices = [p["price"] for p in cluster]
            first_time = min(p["pivot_time"] for p in cluster)
            last_time = max(p["pivot_time"] for p in cluster)
            last_confirmed = max(p["confirmed_at"] for p in cluster)
            tf_set = sorted({p["source_tf"] for p in cluster})
            touches = len(cluster)
            age_days = max((now - last_time).total_seconds() / 86400, 0)
            tf_bonus = 1 if len(tf_set) > 1 else 0
            recency_bonus = 1 if age_days <= 14 else 0
            score = touches + tf_bonus + recency_bonus
            zones.append(
                {
                    "kind": kind,
                    "zone_low": min(prices),
                    "zone_high": max(prices),
                    "zone_mid": float(np.mean(prices)),
                    "pivot_touches": touches,
                    "timeframes": ",".join(tf_set),
                    "first_pivot_time": first_time,
                    "last_pivot_time": last_time,
                    "last_confirmed_at": last_confirmed,
                    "age_days": round(age_days, 1),
                    "score": int(score),
                }
            )

    result = pd.DataFrame(zones)
    if result.empty:
        return result

    result["zone_width"] = result["zone_high"] - result["zone_low"]
    result["quality"] = np.select(
        [result["score"] >= 6, result["score"] >= 4],
        ["STRONG_CANDIDATE", "MODERATE_CANDIDATE"],
        default="WEAK",
    )
    return result.sort_values(["score", "last_pivot_time"], ascending=[False, False])


def candle_signal(m5, m15, h1, h4, zones):
    latest = m5.iloc[-1]
    latest_time = latest["time"]
    close = float(latest["close"])
    open_ = float(latest["open"])
    high = float(latest["high"])
    low = float(latest["low"])
    candle_range = max(high - low, 1e-9)
    body = abs(close - open_)
    lower_wick = min(open_, close) - low
    upper_wick = high - max(open_, close)

    h4row = h4.iloc[-1]
    h1row = h1.iloc[-1]
    h4_bull = h4row["h4_ema50"] > h4row["h4_ema200"]
    h4_bear = h4row["h4_ema50"] < h4row["h4_ema200"]
    h1_bull = h1row["h1_ema20"] > h1row["h1_ema50"]
    h1_bear = h1row["h1_ema20"] < h1row["h1_ema50"]

    recent_m5 = m5.tail(SIGNAL_LOOKBACK_BARS)
    momentum_bull = (
        close > open_
        and body / candle_range >= 0.55
        and close >= float(recent_m5["high"].iloc[:-1].max())
    ) if len(recent_m5) > 1 else False
    momentum_bear = (
        close < open_
        and body / candle_range >= 0.55
        and close <= float(recent_m5["low"].iloc[:-1].min())
    ) if len(recent_m5) > 1 else False

    candidates = []
    for _, zone in zones.iterrows():
        low_zone = float(zone["zone_low"])
        high_zone = float(zone["zone_high"])
        mid = float(zone["zone_mid"])
        atr = float(m5["m5_atr14"].iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            continue
        near = low_zone - 0.20 * atr <= close <= high_zone + 0.20 * atr
        touched = low <= high_zone and high >= low_zone
        if not (near or touched):
            continue

        if zone["kind"] == "SUPPORT":
            rejection = lower_wick >= max(body * 1.2, 0.15 * atr) and close > low_zone
            direction_ok = h4_bull and h1_bull
            confirmation = rejection or momentum_bull
            direction = "LONG"
        else:
            rejection = upper_wick >= max(body * 1.2, 0.15 * atr) and close < high_zone
            direction_ok = h4_bear and h1_bear
            confirmation = rejection or momentum_bear
            direction = "SHORT"

        status = "WAIT"
        reason = "Belum ada konfirmasi candle atau tren timeframe tinggi tidak searah."
        if direction_ok and confirmation:
            status = "WATCH_LONG" if direction == "LONG" else "WATCH_SHORT"
            reason = (
                "Tren H4/H1 searah dan ada rejection/momentum di dekat zona. "
                "Validasi spread, news, dan candle berikutnya sebelum entry."
            )
        elif direction_ok:
            reason = "Tren H4/H1 searah, tetapi tunggu rejection atau momentum candle."

        candidates.append(
            {
                "signal_time_utc": latest_time,
                "direction": direction,
                "status": status,
                "zone_kind": zone["kind"],
                "zone_low": low_zone,
                "zone_high": high_zone,
                "zone_mid": mid,
                "zone_score": int(zone["score"]),
                "zone_quality": zone["quality"],
                "h4_trend": "BULLISH" if h4_bull else "BEARISH" if h4_bear else "MIXED",
                "h1_trend": "BULLISH" if h1_bull else "BEARISH" if h1_bear else "MIXED",
                "candle_rejection": bool(rejection),
                "candle_momentum": bool(momentum_bull if direction == "LONG" else momentum_bear),
                "reason": reason,
            }
        )

    if not candidates:
        return pd.DataFrame(
            [{
                "signal_time_utc": latest_time,
                "direction": "NONE",
                "status": "NO_TRADE",
                "zone_kind": "",
                "zone_low": np.nan,
                "zone_high": np.nan,
                "zone_mid": np.nan,
                "zone_score": 0,
                "zone_quality": "",
                "h4_trend": "BULLISH" if h4_bull else "BEARISH" if h4_bear else "MIXED",
                "h1_trend": "BULLISH" if h1_bull else "BEARISH" if h1_bear else "MIXED",
                "candle_rejection": False,
                "candle_momentum": False,
                "reason": "Harga candle M5 terakhir tidak berada di dekat zona S/R terdeteksi.",
            }]
        )

    result = pd.DataFrame(candidates)
    result["_status_rank"] = result["status"].map(
        {"WATCH_LONG": 0, "WATCH_SHORT": 0, "WAIT": 1, "NO_TRADE": 2}
    ).fillna(3)
    return result.sort_values(
        ["_status_rank", "zone_score"], ascending=[True, False]
    ).drop(columns="_status_rank")


def main():
    print("Memuat data XAUUSD...")
    m5 = add_indicators(load_bars("m5"), "m5")
    m15 = add_indicators(load_bars("m15"), "m15")
    h1 = add_indicators(load_bars("h1"), "h1")
    h4 = add_indicators(load_bars("h4"), "h4")

    # Hanya candle yang sudah selesai yang boleh digunakan.
    now = min(m5["time"].iloc[-1] + pd.Timedelta(minutes=5),
              m15["time"].iloc[-1] + pd.Timedelta(minutes=15),
              h1["time"].iloc[-1] + pd.Timedelta(hours=1),
              h4["time"].iloc[-1] + pd.Timedelta(hours=4))
    current_atr = float(h1["h1_atr14"].iloc[-1])
    pivots = []
    pivots.extend(find_confirmed_pivots(h1, "h1"))
    pivots.extend(find_confirmed_pivots(m15, "m15"))
    zones = cluster_zones(pivots, now, current_atr)

    if zones.empty:
        print("Belum cukup pivot terkonfirmasi untuk membentuk zona.")
        return

    zones = zones[zones["score"] >= MIN_ZONE_SCORE].copy()
    if zones.empty:
        print("Tidak ada zona yang memenuhi skor minimum saat ini.")
        return

    zones["distance_from_price"] = (
        zones["zone_mid"] - float(m5["close"].iloc[-1])
    ).abs()
    zones = zones.sort_values(["distance_from_price", "score"], ascending=[True, False])
    zone_path = REPORTS / "snr_zones_latest.csv"
    zones.to_csv(zone_path, index=False)

    signals = candle_signal(m5, m15, h1, h4, zones)
    signal_path = REPORTS / "snr_signals_latest.csv"
    signals.to_csv(signal_path, index=False)

    print("\n=== KONTEKS TIMEFRAME TINGGI ===")
    print("Candle M5 terakhir:", m5["time"].iloc[-1], "(waktu buka candle UTC)")
    print("Harga penutupan M5:", round(float(m5["close"].iloc[-1]), 3))
    print("H4:", "BULLISH" if h4["h4_ema50"].iloc[-1] > h4["h4_ema200"].iloc[-1]
          else "BEARISH" if h4["h4_ema50"].iloc[-1] < h4["h4_ema200"].iloc[-1] else "MIXED")
    print("H1:", "BULLISH" if h1["h1_ema20"].iloc[-1] > h1["h1_ema50"].iloc[-1]
          else "BEARISH" if h1["h1_ema20"].iloc[-1] < h1["h1_ema50"].iloc[-1] else "MIXED")

    print("\n=== ZONA S/R TERDEKAT ===")
    cols = ["kind", "zone_low", "zone_high", "zone_mid", "pivot_touches",
            "timeframes", "age_days", "score", "quality", "distance_from_price"]
    print(zones[cols].head(12).round(3).to_string(index=False))

    print("\n=== PEMERIKSAAN ENTRY M5 ===")
    print(signals.to_string(index=False))
    print("\nOutput:")
    print(zone_path)
    print(signal_path)
    print("\nIni scanner kandidat, bukan sistem entry otomatis. WATCH tetap perlu cek spread/news dan validasi backtest.")


if __name__ == "__main__":
    main()
