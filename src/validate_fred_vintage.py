from pathlib import Path
import pandas as pd

root = Path.cwd() / "data" / "raw"

files = {
    "DGS10": root / "fred_vintage_DGS10.csv",
    "DFII10": root / "fred_vintage_DFII10.csv",
    "CPIAUCNS": root / "fred_vintage_CPIAUCNS.csv",
}

for series, path in files.items():
    print(f"\n{'=' * 60}")
    print(f"VALIDASI {series}")
    print(f"{'=' * 60}")

    if not path.exists():
        print(f"FILE TIDAK DITEMUKAN: {path}")
        continue

    df = pd.read_csv(path, parse_dates=[
        "observation_date", "vintage_date"
    ])

    df["vintage_value"] = pd.to_numeric(
        df["vintage_value"], errors="coerce"
    )

    print("Jumlah baris:", len(df))
    print("Tanggal observasi:", df["observation_date"].min(),
          "sampai", df["observation_date"].max())
    print("Tanggal vintage:", df["vintage_date"].min(),
          "sampai", df["vintage_date"].max())
    print("Jumlah vintage:", df["vintage_date"].nunique())
    print("Baris tanpa nilai:", df["vintage_value"].isna().sum())

    duplicates = df.duplicated(
        ["observation_date", "vintage_date"]
    ).sum()
    print("Pasangan tanggal duplikat:", duplicates)

    future = df[
        (df["observation_date"] > df["vintage_date"])
        & df["vintage_value"].notna()
    ]
    print(
        "Nilai observasi masa depan yang terisi:",
        len(future)
    )

    print("\nContoh snapshot 12 Januari 2023:")
    sample = df[df["vintage_date"] == "2023-01-01"].head(5)
    if sample.empty:
        print("Tidak ada snapshot tepat pada tanggal tersebut.")
    else:
        print(sample.to_string(index=False))

print("\nValidasi selesai. File sumber tidak diubah.")
