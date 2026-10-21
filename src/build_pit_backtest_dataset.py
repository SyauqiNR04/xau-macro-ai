from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()
MACRO_FILE = ROOT / "data" / "processed" / "macro_score_pit_v1.csv"
GOLD_FILE = ROOT / "data" / "raw" / "xauusd_d1.csv"
OUTPUT = ROOT / "data" / "processed" / "backtest_pit_v1.csv"

for path in [MACRO_FILE, GOLD_FILE]:
    if not path.exists():
        raise SystemExit(f"File tidak ditemukan: {path}")

macro = pd.read_csv(MACRO_FILE, parse_dates=["date"])
gold = pd.read_csv(GOLD_FILE)

# Normalisasi tanggal candle XAUUSD.
date_col = "time" if "time" in gold.columns else "date"
if date_col not in gold.columns:
    raise SystemExit("Kolom tanggal tidak ditemukan di data XAUUSD.")

gold["date"] = pd.to_datetime(
    gold[date_col], errors="coerce", utc=True
).dt.tz_convert(None).dt.normalize()

gold["close"] = pd.to_numeric(gold["close"], errors="coerce")
gold = gold.dropna(subset=["date", "close"])
gold = gold.sort_values("date").drop_duplicates("date").reset_index(drop=True)

macro["date"] = pd.to_datetime(macro["date"], errors="coerce").dt.normalize()
macro = macro.dropna(subset=["date"])
macro = macro.sort_values("date").drop_duplicates("date")

macro_cols = [
    "date",
    "macro_score",
    "macro_bias",
    "change_5obs_DFII10",
    "latest_observation_date_DFII10",
    "real_yield_age_days",
    "cpi_yoy_context",
]
macro = macro[macro_cols].copy()

# Skor bertanggal D berasal dari fitur konservatif yang dilag.
# Cocokkan ke candle D; return dihitung mulai dari close D menuju
# close candle berikutnya, bukan return candle D itu sendiri.
df = gold.merge(macro, on="date", how="left", validate="one_to_one")

for horizon in [1, 3, 5]:
    future_close = df["close"].shift(-horizon)
    df[f"forward_return_{horizon}bar_pct"] = (
        future_close / df["close"] - 1
    ) * 100

# Sinyal hanya valid jika tersedia dan usia observasi yield <= 3 hari.
df["signal_eligible"] = (
    df["macro_score"].notna()
    & df["real_yield_age_days"].between(0, 3)
)

for horizon in [1, 3, 5]:
    ret_col = f"forward_return_{horizon}bar_pct"
    df[f"direction_correct_{horizon}bar"] = np.where(
        df["signal_eligible"]
        & df["macro_score"].ne(0)
        & df[ret_col].notna(),
        (
            np.sign(df["macro_score"])
            == np.sign(df[ret_col])
        ).astype(float),
        np.nan,
    )

df.to_csv(OUTPUT, index=False)

print("=== DATASET BACKTEST PIT V1 ===")
print("File:", OUTPUT)
print("Jumlah candle:", len(df))
print("Rentang:", df["date"].min(), "sampai", df["date"].max())
print("Baris dengan skor:", int(df["macro_score"].notna().sum()))
print("Sinyal eligible (usia yield 0-3 hari):",
      int(df["signal_eligible"].sum()))

print("\n=== EVALUASI DESKRIPTIF ===")
for horizon in [1, 3, 5]:
    ret_col = f"forward_return_{horizon}bar_pct"
    correct_col = f"direction_correct_{horizon}bar"

    valid = df[
        df["signal_eligible"]
        & df["macro_score"].ne(0)
        & df[ret_col].notna()
    ].copy()

    if valid.empty:
        print(f"\n{horizon} candle: tidak ada sampel valid.")
        continue

    print(f"\nHorizon {horizon} candle")
    print("Jumlah sinyal:", len(valid))
    print("Directional accuracy (%):",
          round(valid[correct_col].mean() * 100, 2))
    print("Rata-rata forward return (%):",
          round(valid[ret_col].mean(), 4))
    print("Median forward return (%):",
          round(valid[ret_col].median(), 4))
    print("Bullish:", int((valid["macro_score"] > 0).sum()))
    print("Bearish:", int((valid["macro_score"] < 0).sum()))

print("\nCatatan:")
print("- Return dihitung dari close sinyal ke close masa depan.")
print("- Horizon adalah jumlah candle, bukan jumlah hari kalender.")
print("- Belum memperhitungkan spread, komisi, dan slippage.")
print("- Akurasi arah tidak sama dengan profitabilitas.")
