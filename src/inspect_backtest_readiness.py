import pandas as pd
from pathlib import Path

files = {
    "M5": "data/raw/xauusdm_m5.csv",
    "M15": "data/raw/xauusdm_m15.csv",
    "H1": "data/raw/xauusdm_h1.csv",
    "H4": "data/raw/xauusdm_h4.csv",
    "MACRO": "data/processed/macro_score_pit_v1.csv",
}

for name, filename in files.items():
    print("\n" + "=" * 65)
    print(name)

    df = pd.read_csv(filename)
    time_col = "time" if "time" in df.columns else "date"
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    print("Baris:", len(df))
    print("Rentang:", df[time_col].min(), "hingga", df[time_col].max())

    if name == "MACRO":
        print("\nDistribusi macro_bias:")
        print(df["macro_bias"].value_counts(dropna=False).to_string())

        print("\nKetersediaan faktor:")
        print(df["available_factors"].value_counts(dropna=False).sort_index().to_string())

        print("\nPersentase skor tersedia:")
        print(round(df["macro_score"].notna().mean() * 100, 2), "%")

        print("\nContoh skor terbaru:")
        print(df.tail(10).to_string(index=False))

print("\nAudit awal selesai.")
