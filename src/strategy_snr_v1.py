from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Versi awal: scanner SnR, bukan auto-trader atau backtest.
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
ATR_PERIOD = 14
ZONE_ATR_TOLERANCE = 0.45
MIN_ZONE_SCORE = 2
MAX_ZONES_PER_TF = 12
NEAR_ZONE_ATR = 0.35


def load_candles(filename: str, timeframe_minutes: int) -> pd.DataFrame:
    path = RAW / filename
    if not path.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {path}")

    df = pd.read_csv(path)
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename}: kolom hilang: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    # File MT5 menyimpan waktu buka candle. Buang candle terakhir karena
    # kemungkinan masih berjalan; analisis hanya memakai candle tertutup.
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    if len(df) < max(ATR_PERIOD + 5, PIVOT_LEFT + PIVOT_RIGHT + 5):
        raise ValueError(f"Data {filename} terlalu sedikit untuk analisis.")

    df["available_at"] = df["time"] + pd.Timedelta(minutes=timeframe_minutes)
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = true_range.rolling(ATR_PERIOD).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    return df


def find_pivots(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rows = []
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr"].to_numpy()
    times = df["available_at"].to_numpy()

    # Pivot baru dianggap diketahui setelah PIVOT_RIGHT candle berikutnya
    # selesai. Waktu pivot disimpan sebagai waktu konfirmasi, bukan waktu
    # candle pivot, untuk menghindari penggunaan level sebelum terkonfirmasi.
    for i in range(PIVOT_LEFT, len(df) - PIVOT_RIGHT):
        left = i - PIVOT_LEFT
        right = i + PIVOT_RIGHT + 1
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        is_high = high[i] >= np.max(high[left:right])
        is_low = low[i] <= np.min(low[left:right])

        if is_high:
            rows.append({
                "timeframe": timeframe,
                "kind": "resistance",
                "pivot_time": df.iloc[i]["time"],
                "confirmed_at": pd.Timestamp(times[i + PIVOT_RIGHT]),
                "price": float(high[i]),
                "atr_at_pivot": float(atr[i]),
            })
        if is_low:
            rows.append({
                "timeframe": timeframe,
                "kind": "support",
                "pivot_time": df.iloc[i]["time"],
                "confirmed_at": pd.Timestamp(times[i + PIVOT_RIGHT]),
                "price": float(low[i]),
                "atr_at_pivot": float(atr[i]),
            })

    return pd.DataFrame(rows)


def cluster_zones(pivots: pd.DataFrame, latest_price: float, latest_atr: float) -> pd.DataFrame:
    if pivots.empty:
        return pd.DataFrame()

    zones = []
    for (timeframe, kind), group in pivots.groupby(["timeframe", "kind"]):
        group = group.sort_values("price")
        clusters = []

        for row in group.to_dict("records"):
            tolerance = max(
                float(row["atr_at_pivot"]) * ZONE_ATR_TOLERANCE,
                latest_atr * 0.05,
            )
            target = None
            for cluster in clusters:
                if abs(row["price"] - cluster["center"]) <= tolerance:
                    target = cluster
                    break

            if target is None:
                clusters.append({
                    "center": row["price"],
                    "items": [row],
                })
            else:
                target["items"].append(row)
                target["center"] = float(np.mean([x["price"] for x in target["items"]]))

        for cluster in clusters:
            items = cluster["items"]
            prices = [x["price"] for x in items]
            last_confirmed = max(x["confirmed_at"] for x in items)
            touches = len(items)
            distance_atr = abs(float(np.mean(prices)) - latest_price) / latest_atr
            score = min(touches, 5)
            if distance_atr <= 1:
                score += 1
            if distance_atr <= 2:
                score += 1

            zones.append({
                "timeframe": timeframe,
                "kind": kind,
                "zone_low": min(prices),
                "zone_high": max(prices),
                "zone_mid": float(np.mean(prices)),
                "pivot_count": touches,
                "last_confirmed_at": last_confirmed,
                "distance_atr": round(distance_atr, 3),
                "score": int(score),
            })

    result = pd.DataFrame(zones)
    if result.empty:
        return result

    result = result[result["score"] >= MIN_ZONE_SCORE].copy()
    result = result.sort_values(
        ["timeframe", "distance_atr", "score"],
        ascending=[True, True, False],
    )
    return result.groupby("timeframe", group_keys=False).head(MAX_ZONES_PER_TF).reset_index(drop=True)


def get_trend(h4: pd.DataFrame, h1: pd.DataFrame) -> tuple[str, str]:
    a = h4.iloc[-1]
    b = h1.iloc[-1]

    if a["ema50"] > a["ema200"] and a["close"] > a["ema50"]:
        h4_trend = "BULLISH"
    elif a["ema50"] < a["ema200"] and a["close"] < a["ema50"]:
        h4_trend = "BEARISH"
    else:
        h4_trend = "RANGE / MIXED"

    if b["ema20"] > b["ema50"] and b["close"] > b["ema20"]:
        h1_trend = "BULLISH"
    elif b["ema20"] < b["ema50"] and b["close"] < b["ema20"]:
        h1_trend = "BEARISH"
    else:
        h1_trend = "RANGE / MIXED"

    return h4_trend, h1_trend


def candle_confirmation(m5: pd.DataFrame, m15: pd.DataFrame) -> tuple[str, str]:
    c = m5.iloc[-1]
    prev = m5.iloc[-2]
    candle_range = float(c["high"] - c["low"])
    if candle_range <= 0 or not np.isfinite(c["atr"]) or c["atr"] <= 0:
        return "NONE", "Rentang candle/ATR tidak valid"

    body = abs(float(c["close"] - c["open"]))
    body_ratio = body / candle_range
    bullish = c["close"] > c["open"]
    bearish = c["close"] < c["open"]
    momentum = candle_range >= 0.8 * c["atr"] and body_ratio >= 0.55
    bull_engulf = (
        bullish and prev["close"] < prev["open"]
        and c["close"] >= prev["open"] and c["open"] <= prev["close"]
    )
    bear_engulf = (
        bearish and prev["close"] > prev["open"]
        and c["close"] <= prev["open"] and c["open"] >= prev["close"]
    )

    m15_last = m15.iloc[-1]
    if momentum and bullish:
        return "BULLISH_MOMENTUM", "Candle M5 bullish dengan body relatif besar"
    if momentum and bearish:
        return "BEARISH_MOMENTUM", "Candle M5 bearish dengan body relatif besar"
    if bull_engulf:
        return "BULLISH_ENGULFING", "Pola bullish engulfing M5"
    if bear_engulf:
        return "BEARISH_ENGULFING", "Pola bearish engulfing M5"
    if m15_last["close"] > m15_last["ema20"] and bullish:
        return "WEAK_BULLISH", "M15 di atas EMA20 dan candle M5 bullish"
    if m15_last["close"] < m15_last["ema20"] and bearish:
        return "WEAK_BEARISH", "M15 di bawah EMA20 dan candle M5 bearish"
    return "NONE", "Belum ada konfirmasi candle yang jelas"


def main():
    h4 = load_candles("xauusdm_h4.csv", 240)
    h1 = load_candles("xauusdm_h1.csv", 60)
    m15 = load_candles("xauusdm_m15.csv", 15)
    m5 = load_candles("xauusdm_m5.csv", 5)

    h4_trend, h1_trend = get_trend(h4, h1)
    current = m5.iloc[-1]
    current_price = float(current["close"])
    current_atr = float(current["atr"])

    if not np.isfinite(current_atr) or current_atr <= 0:
        raise RuntimeError("ATR M5 terakhir tidak valid.")

    pivot_frames = [
        find_pivots(h4, "H4"),
        find_pivots(h1, "H1"),
    ]
    pivots = pd.concat(pivot_frames, ignore_index=True) if any(not x.empty for x in pivot_frames) else pd.DataFrame()
    zones = cluster_zones(pivots, current_price, current_atr) if not pivots.empty else pd.DataFrame()

    if not zones.empty:
        zones["distance_price"] = zones["zone_mid"] - current_price
        zones["nearby"] = zones["distance_atr"] <= NEAR_ZONE_ATR
        zones.to_csv(REPORTS / "snr_zones.csv", index=False)

    confirmation, confirmation_note = candle_confirmation(m5, m15)

    nearby = pd.DataFrame()
    if not zones.empty:
        nearby = zones[zones["distance_atr"] <= 2].sort_values("distance_atr")

    aligned = h4_trend == h1_trend and h4_trend in {"BULLISH", "BEARISH"}
    direction = "WAIT"
    reason = "NO TRADE: tren H4/H1 belum searah atau belum ada konfirmasi."

    if aligned and not nearby.empty:
        supports = nearby[nearby["kind"] == "support"]
        resistances = nearby[nearby["kind"] == "resistance"]
        if h4_trend == "BULLISH" and not supports.empty and confirmation.startswith("BULLISH"):
            direction = "POTENTIAL LONG"
            reason = "Tren H4/H1 bullish, ada zona support relatif dekat, dan candle M5 bullish."
        elif h4_trend == "BEARISH" and not resistances.empty and confirmation.startswith("BEARISH"):
            direction = "POTENTIAL SHORT"
            reason = "Tren H4/H1 bearish, ada zona resistance relatif dekat, dan candle M5 bearish."

    summary = pd.DataFrame([{
        "analysis_time_utc": current["available_at"],
        "price": current_price,
        "m5_atr": current_atr,
        "h4_trend": h4_trend,
        "h1_trend": h1_trend,
        "trend_aligned": aligned,
        "candle_confirmation": confirmation,
        "confirmation_note": confirmation_note,
        "assessment": direction,
        "reason": reason,
    }])
    summary.to_csv(REPORTS / "snr_current_analysis.csv", index=False)

    print("=== XAUUSD SNR SCANNER V1 ===")
    print("Waktu analisis UTC:", current["available_at"])
    print("Harga acuan M5:", round(current_price, 3))
    print("Tren H4:", h4_trend, "| Tren H1:", h1_trend)
    print("Konfirmasi candle:", confirmation, "-", confirmation_note)
    print("Penilaian:", direction)
    print("Alasan:", reason)
    print("\nZona SnR terdekat (maksimum 10):")
    if nearby.empty:
        print("Belum ditemukan zona H4/H1 yang memenuhi filter.")
    else:
        cols = ["timeframe", "kind", "zone_low", "zone_high", "zone_mid", "pivot_count", "distance_atr", "score"]
        print(nearby[cols].head(10).round(3).to_string(index=False))

    print("\nOutput:")
    print(REPORTS / "snr_current_analysis.csv")
    if not zones.empty:
        print(REPORTS / "snr_zones.csv")
    print("\nCatatan: scanner ini hanya memberi kandidat analisis, bukan perintah entry.")
    print("Belum menghitung SL/TP, spread aktual, atau hasil backtest.")


if __name__ == "__main__":
    main()
