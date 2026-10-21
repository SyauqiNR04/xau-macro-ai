import pandas as pd
import numpy as np

path = "data/raw/xauusdm_m5.csv"
df = pd.read_csv(path)
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values("time").reset_index(drop=True)

df["previous_close"] = df["close"].shift(1)
df["gap_minutes"] = (
    df["time"] - df["time"].shift(1)
).dt.total_seconds() / 60

df["gap_price"] = df["open"] - df["previous_close"]
df["gap_abs"] = df["gap_price"].abs()

print("=== DISTRIBUSI JEDA ANTAR-CANDLE ===")
print(df["gap_minutes"].describe(percentiles=[.5, .9, .95, .99]).to_string())

print("\n=== JEDA LEBIH DARI 10 MENIT ===")
gaps = df[df["gap_minutes"] > 10].copy()
print("Jumlah:", len(gaps))
print("Gap harga absolut:")
print(gaps["gap_abs"].describe(percentiles=[.5, .9, .95, .99]).round(4).to_string())

print("\n=== 20 GAP TERBESAR ===")
cols = ["time", "previous_close", "open", "gap_price", "gap_minutes"]
print(
    gaps.reindex(gaps["gap_abs"].sort_values(ascending=False).index)[cols]
    .head(20)
    .to_string(index=False)
)

print("\nAudit gap selesai.")
