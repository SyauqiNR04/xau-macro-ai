from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
SOURCE = ROOT / "data" / "processed" / "fred_pit_daily_features.csv"
OUTPUT = ROOT / "data" / "processed" / "fred_pit_daily_features_conservative.csv"

if not SOURCE.exists():
    raise SystemExit(f"File tidak ditemukan: {SOURCE}")

df = pd.read_csv(SOURCE, parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
df = (
    df.dropna(subset=["date"])
    .sort_values("date")
    .drop_duplicates("date")
    .set_index("date")
    .asfreq("D")
)

# Catat tanggal snapshot asli sebelum menggeser fitur.
df["feature_vintage_date"] = df.index

drop_cols = [
    "change_1obs_CPIAUCNS",
    "change_5obs_CPIAUCNS",
    "yoy_pct_DGS10",
    "yoy_pct_DFII10",
]
df = df.drop(columns=drop_cols, errors="ignore")

cpi_cols = [
    "latest_observation_date_CPIAUCNS",
    "latest_value_CPIAUCNS",
    "yoy_pct_CPIAUCNS",
]
for col in cpi_cols:
    if col in df.columns:
        df[col] = df[col].ffill()

df = df.shift(1).reset_index()
df.to_csv(OUTPUT, index=False)

print("=== HASIL ===")
print("File:", OUTPUT)
print("Jumlah baris:", len(df))
print("Rentang:", df["date"].min(), "sampai", df["date"].max())

print("\n=== AUDIT TANGGAL ASAL FITUR ===")
print(df[["date", "feature_vintage_date"]].tail(10).to_string(index=False))

valid = df.dropna(subset=["feature_vintage_date"]).copy()
valid["calendar_lag_days"] = (
    valid["date"] - valid["feature_vintage_date"]
).dt.days

print("\nDistribusi jarak kalender:")
print(valid["calendar_lag_days"].value_counts().sort_index().to_string())

print("\nNilai kosong:")
print(df.isna().sum().to_string())
