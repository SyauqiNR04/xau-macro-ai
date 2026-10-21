
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "xauusd_d1.csv"

# Sesuaikan dengan nama simbol persis di Market Watch broker.
SYMBOL = "XAUUSD"

# Mulai 2015 agar tersedia rentang historis yang cukup untuk pengujian.
DATE_FROM = datetime(2015, 1, 1, tzinfo=timezone.utc)
DATE_TO = datetime.now(timezone.utc)


def main():
    if not mt5.initialize():
        raise RuntimeError(
            f"Gagal terhubung ke MT5: {mt5.last_error()}. "
            "Pastikan terminal MT5 terpasang dan dapat dibuka."
        )

    try:
        symbol_info = mt5.symbol_info(SYMBOL)

        if symbol_info is None:
            candidates = mt5.symbols_get("*XAU*")
            names = [item.name for item in candidates] if candidates else []
            raise ValueError(
                f"Simbol {SYMBOL} tidak ditemukan. "
                f"Simbol yang mengandung XAU: {names[:30]}. "
                "Ubah SYMBOL sesuai nama broker."
            )

        if not symbol_info.visible:
            if not mt5.symbol_select(SYMBOL, True):
                raise RuntimeError(
                    f"Gagal mengaktifkan simbol {SYMBOL}: "
                    f"{mt5.last_error()}"
                )

        rates = mt5.copy_rates_range(
            SYMBOL,
            mt5.TIMEFRAME_D1,
            DATE_FROM,
            DATE_TO,
        )

        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"Tidak ada data candle. Detail MT5: {mt5.last_error()}. "
                "Periksa koneksi, simbol, dan histori chart."
            )

        df = pd.DataFrame(rates)

        df["time_utc"] = pd.to_datetime(
            df["time"], unit="s", utc=True
        )
        df["date"] = df["time_utc"].dt.strftime("%Y-%m-%d")

        df = df.rename(
            columns={
                "tick_volume": "tick_volume",
                "real_volume": "real_volume",
            }
        )

        columns = [
            "date",
            "time_utc",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]

        df = df[columns].sort_values("time_utc")
        df = df.drop_duplicates(subset=["time_utc"], keep="last")

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)

        print("\n=== XAUUSD D1 BERHASIL DIAMBIL ===")
        print(f"Simbol: {SYMBOL}")
        print(f"Jumlah candle: {len(df)}")
        print(f"Tanggal awal: {df['date'].min()}")
        print(f"Tanggal akhir: {df['date'].max()}")
        print(f"File: {OUTPUT_FILE}")
        print("\n5 candle terbaru:")
        print(df.tail(5).to_string(index=False))

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()

from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "xauusd_d1.csv"

# Sesuaikan dengan nama simbol persis di Market Watch broker.
SYMBOL = "XAUUSD"

# Mulai 2015 agar tersedia rentang historis yang cukup untuk pengujian.
DATE_FROM = datetime(2015, 1, 1, tzinfo=timezone.utc)
DATE_TO = datetime.now(timezone.utc)


def main():
    if not mt5.initialize():
        raise RuntimeError(
            f"Gagal terhubung ke MT5: {mt5.last_error()}. "
            "Pastikan terminal MT5 terpasang dan dapat dibuka."
        )

    try:
        symbol_info = mt5.symbol_info(SYMBOL)

        if symbol_info is None:
            candidates = mt5.symbols_get("*XAU*")
            names = [item.name for item in candidates] if candidates else []
            raise ValueError(
                f"Simbol {SYMBOL} tidak ditemukan. "
                f"Simbol yang mengandung XAU: {names[:30]}. "
                "Ubah SYMBOL sesuai nama broker."
            )

        if not symbol_info.visible:
            if not mt5.symbol_select(SYMBOL, True):
                raise RuntimeError(
                    f"Gagal mengaktifkan simbol {SYMBOL}: "
                    f"{mt5.last_error()}"
                )

        rates = mt5.copy_rates_range(
            SYMBOL,
            mt5.TIMEFRAME_D1,
            DATE_FROM,
            DATE_TO,
        )

        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"Tidak ada data candle. Detail MT5: {mt5.last_error()}. "
                "Periksa koneksi, simbol, dan histori chart."
            )

        df = pd.DataFrame(rates)

        df["time_utc"] = pd.to_datetime(
            df["time"], unit="s", utc=True
        )
        df["date"] = df["time_utc"].dt.strftime("%Y-%m-%d")

        df = df.rename(
            columns={
                "tick_volume": "tick_volume",
                "real_volume": "real_volume",
            }
        )

        columns = [
            "date",
            "time_utc",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]

        df = df[columns].sort_values("time_utc")
        df = df.drop_duplicates(subset=["time_utc"], keep="last")

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)

        print("\n=== XAUUSD D1 BERHASIL DIAMBIL ===")
        print(f"Simbol: {SYMBOL}")
        print(f"Jumlah candle: {len(df)}")
        print(f"Tanggal awal: {df['date'].min()}")
        print(f"Tanggal akhir: {df['date'].max()}")
        print(f"File: {OUTPUT_FILE}")
        print("\n5 candle terbaru:")
        print(df.tail(5).to_string(index=False))

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
