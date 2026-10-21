from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

POINT = 0.001
ATR_MULT = 1.5
TARGET_R = 2.0
MAX_HOLD_BARS = 48
SLIPPAGE_POINTS = 20
OOS_START = pd.Timestamp("2024-01-01", tz="UTC")


def load_ohlc(filename, minutes, prefix):
    df = pd.read_csv(RAW / filename)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "spread"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["available_at"] = df["time"] + pd.Timedelta(minutes=minutes)
    df[f"{prefix}_ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df[f"{prefix}_ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df[f"{prefix}_ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df[f"{prefix}_atr14"] = tr.rolling(14).mean()

    keep = [
        "available_at", "open", "high", "low", "close",
        f"{prefix}_ema20", f"{prefix}_ema50",
        f"{prefix}_ema200", f"{prefix}_atr14"
    ]
    return df[keep].copy()


def merge_latest(left, right, prefix):
    right = right.copy()
    rename = {
        c: f"{prefix}_{c}"
        for c in right.columns if c != "available_at"
    }
    right = right.rename(columns=rename)

    return pd.merge_asof(
        left.sort_values("signal_time"),
        right.sort_values("available_at"),
        left_on="signal_time",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True
    ).drop(columns=["available_at"])


def build_dataset():
    m5 = pd.read_csv(RAW / "xauusdm_m5.csv")
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    m5["signal_time"] = m5["time"] + pd.Timedelta(minutes=5)
    m5["spread"] = pd.to_numeric(m5["spread"], errors="coerce").fillna(0)
    m5["m5_prev_high"] = m5["high"].shift(1)
    m5["m5_prev_low"] = m5["low"].shift(1)

    # Indikator timeframe tinggi hanya tersedia setelah candle ditutup.
    h4 = load_ohlc("xauusdm_h4.csv", 240, "h4")
    h1 = load_ohlc("xauusdm_h1.csv", 60, "h1")
    m15 = load_ohlc("xauusdm_m15.csv", 15, "m15")

    df = merge_latest(m5, h4, "h4")
    df = merge_latest(df, h1, "h1")
    df = merge_latest(df, m15, "m15")

    # Macro score tanggal D baru dianggap tersedia mulai tanggal D+1 UTC.
    # Ini adalah aturan konservatif untuk menghindari penggunaan skor hari
    # yang sama sebelum dipastikan tersedia.
    macro_path = PROCESSED / "macro_score_pit_v1.csv"
    macro = pd.read_csv(macro_path)
    macro["date"] = pd.to_datetime(macro["date"], utc=True)
    macro["available_at"] = macro["date"] + pd.Timedelta(days=1)
    macro["macro_score"] = pd.to_numeric(macro["macro_score"], errors="coerce")
    macro["available_factors"] = pd.to_numeric(
        macro["available_factors"], errors="coerce"
    ).fillna(0)

    macro = macro[[
        "available_at", "macro_score", "macro_bias", "available_factors"
    ]].sort_values("available_at")

    df = pd.merge_asof(
        df.sort_values("signal_time"),
        macro,
        left_on="signal_time",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True
    ).drop(columns=["available_at"])

    # Sinyal menggunakan candle M5 yang telah ditutup.
    # Entry dilakukan pada open candle M5 berikutnya.
    long_setup = (
        (df["h4_h4_ema50"] > df["h4_h4_ema200"]) &
        (df["h1_h1_ema20"] > df["h1_h1_ema50"]) &
        (df["m15_close"] > df["m15_m15_ema20"]) &
        (df["m15_low"] <=
         df["m15_m15_ema20"] + 0.25 * df["m15_m15_atr14"]) &
        (df["close"] > df["m5_prev_high"])
    )

    short_setup = (
        (df["h4_h4_ema50"] < df["h4_h4_ema200"]) &
        (df["h1_h1_ema20"] < df["h1_h1_ema50"]) &
        (df["m15_close"] < df["m15_m15_ema20"]) &
        (df["m15_high"] >=
         df["m15_m15_ema20"] - 0.25 * df["m15_m15_atr14"]) &
        (df["close"] < df["m5_prev_low"])
    )

    valid = df["m15_m15_atr14"].notna()

    df["signal"] = np.select(
        [long_setup & valid, short_setup & valid],
        [1, -1],
        default=0
    )

    df["macro_signal"] = np.select(
        [
            df["macro_bias"].eq("bullish_gold"),
            df["macro_bias"].eq("bearish_gold")
        ],
        [1, -1],
        default=0
    )

    df["entry_signal"] = df["signal"].shift(1).fillna(0).astype(int)
    df["entry_atr"] = df["m15_m15_atr14"].shift(1)
    df["entry_macro_signal"] = df["macro_signal"].shift(1).fillna(0).astype(int)
    df["entry_macro_score"] = df["macro_score"].shift(1)
    df["entry_macro_factors"] = df["available_factors"].shift(1).fillna(0)

    return df


def simulate(df, use_macro):
    trades = []
    position = None
    slip = SLIPPAGE_POINTS * POINT

    for i in range(1, len(df)):
        row = df.iloc[i]
        signal = int(row["entry_signal"])

        # Entry pada open candle setelah candle sinyal selesai.
        if position is None:
            if signal == 0 or pd.isna(row["entry_atr"]):
                continue

            if use_macro:
                macro_dir = int(row["entry_macro_signal"])
                if macro_dir != signal or row["entry_macro_factors"] < 1:
                    continue

            risk = float(row["entry_atr"]) * ATR_MULT
            if not np.isfinite(risk) or risk <= 0:
                continue

            spread = max(float(row["spread"]), 0) * POINT
            open_price = float(row["open"])

            if signal == 1:
                entry = open_price + spread + slip
                stop = entry - risk
                target = entry + TARGET_R * risk
            else:
                # OHLC diasumsikan harga bid; short masuk di bid.
                entry = open_price - slip
                stop = entry + risk
                target = entry - TARGET_R * risk

            position = {
                "direction": signal,
                "entry": entry,
                "stop": stop,
                "target": target,
                "risk": risk,
                "entry_time": row["time"],
                "entry_index": i,
                "macro_score": row["entry_macro_score"],
            }
            continue

        direction = position["direction"]
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        spread = max(float(row["spread"]), 0) * POINT

        stop = position["stop"]
        target = position["target"]
        reason = None
        exit_price = None

        if direction == 1:
            # Long keluar di bid. Jika open melewati SL, gunakan open,
            # bukan harga SL yang sudah tidak tersedia.
            if open_price <= stop:
                exit_price = open_price - slip
                reason = "SL_GAP"
            elif open_price >= target:
                # Target limit diasumsikan terisi pada target, konservatif.
                exit_price = target - slip
                reason = "TP_GAP"
            else:
                stop_hit = low <= stop
                target_hit = high >= target

                # OHLC tidak memberi urutan intrabar; pilih SL bila keduanya.
                if stop_hit:
                    exit_price = stop - slip
                    reason = "SL"
                elif target_hit:
                    exit_price = target - slip
                    reason = "TP"

        else:
            # Short keluar di ask.
            ask_open = open_price + spread
            ask_high = high + spread
            ask_low = low + spread

            if ask_open >= stop:
                exit_price = ask_open + slip
                reason = "SL_GAP"
            elif ask_open <= target:
                exit_price = target + slip
                reason = "TP_GAP"
            else:
                stop_hit = ask_high >= stop
                target_hit = ask_low <= target

                if stop_hit:
                    exit_price = stop + slip
                    reason = "SL"
                elif target_hit:
                    exit_price = target + slip
                    reason = "TP"

        # Batas holding 48 candle yang benar-benar tersedia.
        if reason is None and i - position["entry_index"] >= MAX_HOLD_BARS:
            if direction == 1:
                exit_price = close - slip
                pnl = exit_price - position["entry"]
            else:
                exit_price = close + spread + slip
                pnl = position["entry"] - exit_price
            reason = "TIME"
        else:
            if reason is None:
                continue

            if direction == 1:
                pnl = exit_price - position["entry"]
            else:
                pnl = position["entry"] - exit_price

        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": row["time"],
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": position["entry"],
            "exit_price": exit_price,
            "stop": stop,
            "target": target,
            "exit_reason": reason,
            "pnl_price": pnl,
            "R": pnl / position["risk"],
            "macro_score": position["macro_score"],
        })

        position = None

    return pd.DataFrame(trades)


def summarize(trades, label):
    if trades.empty:
        return {
            "strategy": label, "trades": 0, "win_rate_pct": np.nan,
            "expectancy_R": np.nan, "profit_factor": np.nan,
            "max_drawdown_R": np.nan, "net_R": 0.0
        }

    r = trades["R"].astype(float)
    equity = r.cumsum()
    drawdown = equity - equity.cummax()
    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()

    return {
        "strategy": label,
        "trades": len(trades),
        "win_rate_pct": round((r > 0).mean() * 100, 2),
        "expectancy_R": round(r.mean(), 4),
        "profit_factor": round(
            gross_profit / gross_loss, 4
        ) if gross_loss > 0 else np.inf,
        "max_drawdown_R": round(drawdown.min(), 4),
        "net_R": round(r.sum(), 4)
    }


def main():
    print("Membaca dan menyelaraskan data...")
    df = build_dataset()

    print("Jumlah candle:", len(df))
    print("Rentang:", df["time"].min(), "sampai", df["time"].max())
    print("Sinyal teknikal LONG:", int((df["signal"] == 1).sum()))
    print("Sinyal teknikal SHORT:", int((df["signal"] == -1).sum()))
    print("Candle dengan macro bias tersedia:",
          int(df["macro_signal"].ne(0).sum()))

    technical = simulate(df, use_macro=False)
    hybrid = simulate(df, use_macro=True)

    for name, trades in [
        ("technical_only", technical),
        ("technical_macro", hybrid)
    ]:
        trades.to_csv(OUT / f"{name}_trades.csv", index=False)

    results = []
    for label, trades in [
        ("Technical only", technical),
        ("Technical + macro", hybrid)
    ]:
        results.append(summarize(trades, label))

        if not trades.empty:
            oos = trades[trades["entry_time"] >= OOS_START]
        else:
            oos = trades

        results.append(summarize(oos, label + " | OOS 2024+"))

    report = pd.DataFrame(results)
    report.to_csv(OUT / "backtest_v1_summary.csv", index=False)

    print("\n=== HASIL BACKTEST V1 ===")
    print(report.to_string(index=False))

    for label, trades in [
        ("Technical only", technical),
        ("Technical + macro", hybrid)
    ]:
        if trades.empty:
            continue
        trades["year"] = pd.to_datetime(trades["entry_time"]).dt.year
        yearly = trades.groupby("year")["R"].agg(
            trades="count", expectancy_R="mean", net_R="sum"
        ).round(4)
        print(f"\n=== HASIL TAHUNAN: {label} ===")
        print(yearly.to_string())

    print("\nFile output:")
    print(OUT / "backtest_v1_summary.csv")
    print(OUT / "technical_only_trades.csv")
    print(OUT / "technical_macro_trades.csv")
    print("\nCatatan: ini pengujian awal, bukan bukti profitabilitas.")


if __name__ == "__main__":
    main()
