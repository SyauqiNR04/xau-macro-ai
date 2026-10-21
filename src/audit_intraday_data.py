from pathlib import Path
import pandas as pd

FILES = {
    "M5": ("data/raw/xauusdm_m5.csv", 5),
    "M15": ("data/raw/xauusdm_m15.csv", 15),
    "H1": ("data/raw/xauusdm_h1.csv", 60),
    "H4": ("data/raw/xauusdm_h4.csv", 240),
}

REQUIRED = {"time", "open", "high", "low", "close"}

for tf, (filename, minutes) in FILES.items():
    print(f"\n{'=' * 65}")
    print(f"AUDIT {tf}: {filename}")

    path = Path(filename)
    if not path.exists():
        print("ERROR: File tidak ditemukan")
        continue

    df = pd.read_csv(path)
    missing_cols = REQUIRED - set(df.columns)

    if missing_cols:
        print("ERROR: Kolom wajib tidak ditemukan:", sorted(missing_cols))
        print("Kolom tersedia:", list(df.columns))
        continue

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    numeric = ["open", "high", "low", "close"]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_time = int(df["time"].isna().sum())
    invalid_price = int(df[numeric].isna().any(axis=1).sum())
    duplicates = int(df["time"].duplicated().sum())

    df = df.sort_values("time").reset_index(drop=True)
    valid = df.dropna(subset=["time"] + numeric).copy()

    bad_ohlc = (
        (valid["high"] < valid["low"]) |
        (valid["high"] < valid[["open", "close"]].max(axis=1)) |
        (valid["low"] > valid[["open", "close"]].min(axis=1)) |
        (valid[numeric] <= 0).any(axis=1)
    )

    diffs = valid["time"].diff().dt.total_seconds().div(60)
    gaps = diffs[diffs > minutes * 1.5]

    print("Jumlah baris:", len(df))
    print("Awal UTC:", df["time"].min())
    print("Akhir UTC:", df["time"].max())
    print("Timestamp invalid:", invalid_time)
    print("Harga kosong/invalid:", invalid_price)
    print("Timestamp duplikat:", duplicates)
    print("Bar dengan OHLC tidak valid:", int(bad_ohlc.sum()))
    print("Jeda lebih dari 1.5x timeframe:", len(gaps))

    if not gaps.empty:
        gap_info = pd.DataFrame({
            "sebelum_utc": valid.loc[gaps.index - 1, "time"].values,
            "sesudah_utc": valid.loc[gaps.index, "time"].values,
            "jeda_menit": diffs.loc[gaps.index].values,
        })
        print("\n10 jeda terbesar:")
        print(gap_info.sort_values("jeda_menit", ascending=False)
              .head(10).to_string(index=False))

    print("\nJumlah candle per tahun:")
    print(valid.groupby(valid["time"].dt.year).size().to_string())

    if "spread" in df.columns:
        spread = pd.to_numeric(df["spread"], errors="coerce")
        print("\nSpread candle tersimpan (points):")
        print(spread.describe(percentiles=[0.5, 0.9, 0.95]).to_string())

print("\nAudit selesai. File sumber tidak diubah.")
