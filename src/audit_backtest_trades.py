from pathlib import Path
import pandas as pd
import numpy as np

root = Path("reports")

for name in ["technical_only", "technical_macro"]:
    path = root / f"{name}_trades.csv"
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])

    print("\n" + "=" * 72)
    print("STRATEGI:", name)
    print("Jumlah transaksi:", len(df))

    if df.empty:
        continue

    print("\n--- Per arah transaksi ---")
    print(df.groupby("direction")["R"].agg(
        trades="count",
        expectancy_R="mean",
        median_R="median",
        net_R="sum",
        win_rate=lambda x: (x > 0).mean() * 100
    ).round(4).to_string())

    print("\n--- Per alasan keluar ---")
    print(df.groupby("exit_reason")["R"].agg(
        trades="count",
        expectancy_R="mean",
        net_R="sum",
        win_rate=lambda x: (x > 0).mean() * 100
    ).round(4).to_string())

    df["year"] = df["entry_time"].dt.year
    print("\n--- Per tahun ---")
    print(df.groupby("year")["R"].agg(
        trades="count",
        expectancy_R="mean",
        net_R="sum"
    ).round(4).to_string())

    print("\n--- Distribusi R ---")
    print(df["R"].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).round(4).to_string())

    print("\n--- Durasi posisi (menit) ---")
    duration = (
        df["exit_time"] - df["entry_time"]
    ).dt.total_seconds() / 60
    print(duration.describe().round(2).to_string())

    print("\n--- Nilai kosong ---")
    print(df.isna().sum().to_string())

print("\nAudit transaksi selesai.")
