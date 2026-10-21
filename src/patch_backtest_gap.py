from pathlib import Path
import re

path = Path("src/backtest_intraday_v1.py")
source = path.read_text(encoding="utf-8")

backup = path.with_name("backtest_intraday_v1_before_gapfix.py")
backup.write_text(source, encoding="utf-8")

new_function = r'''
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
'''

pattern = r'def simulate\(df, use_macro\):.*?(?=\ndef summarize\()'
updated, count = re.subn(
    pattern,
    new_function.strip() + "\n\n",
    source,
    count=1,
    flags=re.S
)

if count != 1:
    raise RuntimeError(
        "Fungsi simulate() tidak ditemukan atau format berbeda. "
        "File asli tidak diubah."
    )

path.write_text(updated, encoding="utf-8")
print("Fungsi simulasi diperbarui.")
print("Cadangan:", backup.resolve())
