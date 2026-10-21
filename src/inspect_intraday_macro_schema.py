from pathlib import Path
import pandas as pd

files = [
    "data/raw/xauusdm_m5.csv",
    "data/raw/xauusdm_m15.csv",
    "data/raw/xauusdm_h1.csv",
    "data/raw/xauusdm_h4.csv",
    "data/processed/fred_pit_daily_features_conservative.csv",
    "data/processed/macro_score_pit_v1.csv",
]

for filename in files:
    path = Path(filename)
    print("\n" + "=" * 75)
    print("FILE:", filename)

    if not path.exists():
        print("TIDAK DITEMUKAN")
        continue

    df = pd.read_csv(path, nrows=5)
    print("Kolom:", list(df.columns))
    print("Contoh data:")
    print(df.head(3).to_string(index=False))

    # Baca hanya kolom waktu yang namanya umum, jika ada
    time_cols = [
        c for c in df.columns
        if c.lower() in {"time", "date", "datetime", "timestamp"}
    ]
    print("Kandidat kolom waktu:", time_cols)

print("\nPemeriksaan skema selesai.")
