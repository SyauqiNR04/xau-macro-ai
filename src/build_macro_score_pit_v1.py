from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SOURCE = ROOT / "data" / "processed" / "fred_pit_daily_features_conservative.csv"
OUTPUT = ROOT / "data" / "processed" / "macro_score_pit_v1.csv"

if not SOURCE.exists():
    raise SystemExit(f"File sumber tidak ditemukan: {SOURCE}")

df = pd.read_csv(SOURCE, parse_dates=["date"])
df = df.sort_values("date").drop_duplicates("date").copy()

required = [
    "date",
    "latest_observation_date_DFII10",
    "change_5obs_DFII10",
    "yoy_pct_CPIAUCNS",
]
missing = [col for col in required if col not in df.columns]
if missing:
    raise SystemExit(f"Kolom wajib tidak ditemukan: {missing}")

df["latest_observation_date_DFII10"] = pd.to_datetime(
    df["latest_observation_date_DFII10"], errors="coerce"
)

# Umur data real yield yang digunakan pada setiap tanggal skor.
df["real_yield_age_days"] = (
    df["date"] - df["latest_observation_date_DFII10"]
).dt.days

# Perubahan real yield turun = mendukung emas.
# Perubahan real yield naik = menekan emas.
change = pd.to_numeric(df["change_5obs_DFII10"], errors="coerce")

df["real_yield_signal"] = np.select(
    [change < 0, change > 0],
    [1, -1],
    default=0,
).astype(float)

# Jika perubahan yield belum tersedia, skor juga belum tersedia.
df.loc[change.isna(), "real_yield_signal"] = np.nan

df["macro_score"] = df["real_yield_signal"] * 100

df["macro_bias"] = np.select(
    [
        df["macro_score"] > 0,
        df["macro_score"] < 0,
        df["macro_score"] == 0,
    ],
    ["bullish_gold", "bearish_gold", "neutral"],
    default="unavailable",
)

df.loc[df["macro_score"].isna(), "macro_bias"] = "unavailable"

df["available_factors"] = df["macro_score"].notna().astype(int)
df["score_method"] = "DFII10_change_5obs_only"
df["cpi_yoy_context"] = pd.to_numeric(
    df["yoy_pct_CPIAUCNS"], errors="coerce"
)

columns = [
    "date",
    "macro_score",
    "macro_bias",
    "available_factors",
    "change_5obs_DFII10",
    "latest_observation_date_DFII10",
    "real_yield_age_days",
    "cpi_yoy_context",
    "score_method",
]

result = df[columns].copy()
result.to_csv(OUTPUT, index=False)

print("=== MACRO SCORE PIT V1 ===")
print(f"File: {OUTPUT}")
print(f"Jumlah baris: {len(result)}")
print(f"Rentang: {result['date'].min()} sampai {result['date'].max()}")

print("\nDistribusi bias:")
print(result["macro_bias"].value_counts(dropna=False).to_string())

print("\nNilai kosong:")
print(result.isna().sum().to_string())

print("\n5 baris terakhir:")
print(result.tail(5).to_string(index=False))

print("\nCatatan:")
print("- Skor hanya menggunakan perubahan real yield DFII10.")
print("- CPI hanya menjadi konteks, bukan faktor skor.")
print("- Skor +100/-100 adalah aturan heuristik, bukan probabilitas.")
print("- Belum memasukkan indeks dolar atau biaya transaksi.")
