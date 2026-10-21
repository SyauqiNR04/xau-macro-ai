from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

TERMINAL = r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
SYMBOL = "XAUUSDm"
START = datetime(2020, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}

if not mt5.initialize(TERMINAL, timeout=60000):
    raise RuntimeError(f"MT5 initialize gagal: {mt5.last_error()}")

try:
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        raise RuntimeError(f"Simbol {SYMBOL} tidak ditemukan.")

    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"Gagal memilih {SYMBOL}: {mt5.last_error()}")

    print("Simbol:", SYMBOL, "-", info.description)
    print("Periode:", START, "hingga", END)

    for tf_name, tf_value in TIMEFRAMES.items():
        print(f"\nMengunduh timeframe {tf_name}...")
        chunks = []
        cursor = START
        chunk_days = 30 if tf_name in ("M5", "M15") else 180

        while cursor < END:
            next_cursor = min(
                cursor + timedelta(days=chunk_days), END
            )

            rates = mt5.copy_rates_range(
                SYMBOL, tf_value, cursor, next_cursor
            )

            if rates is not None and len(rates) > 0:
                chunks.append(pd.DataFrame(rates))

            cursor = next_cursor

        if not chunks:
            print(f"TIDAK ADA DATA untuk {tf_name}. Error:", mt5.last_error())
            continue

        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates(subset="time").sort_values("time")
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        path = OUT / f"xauusdm_{tf_name.lower()}.csv"
        df.to_csv(path, index=False)

        print("File:", path)
        print("Jumlah candle:", len(df))
        print("Awal data UTC:", df["time"].min())
        print("Akhir data UTC:", df["time"].max())
        print("Duplikat waktu:", df["time"].duplicated().sum())
        print("Ukuran file (KB):", round(path.stat().st_size / 1024, 1))

finally:
    mt5.shutdown()

print("\nSelesai. Periksa cakupan tanggal sebelum backtest.")
