
from pathlib import Path
import pandas as pd

path = Path("data/raw/xauusd_d1.csv")

if not path.exists() or path.stat().st_size == 0:
    raise SystemExit(f"File tidak ada atau kosong: {path}")

df = pd.read_csv(path)

print("Kolom:", df.columns.tolist())
print("Jumlah baris:", len(df))
print(df.head(3).to_string(index=False))
print(df.tail(3).to_string(index=False))

# Cocokkan nama kolom secara case-insensitive.
colmap = {str(c).strip().lower(): c for c in df.columns}
required = ["open", "high", "low", "close"]
missing = [c for c in required if c not in colmap]

if missing:
    raise SystemExit(
        f"Kolom OHLC belum ditemukan: {missing}. "
        "Periksa nama kolom di output di atas."
    )

for c in required:
    df[colmap[c]] = pd.to_numeric(df[colmap[c]], errors="coerce")

ohlc = df[[colmap[c] for c in required]]
print("\nNilai OHLC kosong:", int(ohlc.isna().any(axis=1).sum()))
print("Duplikat baris:", int(df.duplicated().sum()))

valid = (
    (df[colmap["high"]] >= df[colmap["low"]])
    & (df[colmap["high"]] >= df[colmap["open"]])
    & (df[colmap["high"]] >= df[colmap["close"]])
    & (df[colmap["low"]] <= df[colmap["open"]])
    & (df[colmap["low"]] <= df[colmap["close"]])
)
print("Baris OHLC valid:", int(valid.sum()), "/", len(df))

print("\nAudit selesai. Belum ada backtest yang dijalankan.")
