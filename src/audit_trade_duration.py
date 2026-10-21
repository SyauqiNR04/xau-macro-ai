from pathlib import Path
import pandas as pd

files = [
    "reports/technical_only_trades.csv",
    "reports/technical_macro_trades.csv",
]

for filename in files:
    path = Path(filename)
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])

    df["elapsed_minutes"] = (
        df["exit_time"] - df["entry_time"]
    ).dt.total_seconds() / 60

    print("\n" + "=" * 70)
    print("FILE:", filename)
    print("Jumlah transaksi:", len(df))
    print("\nDurasi kalender (menit):")
    print(df["elapsed_minutes"].describe().round(2).to_string())

    print("\nTransaksi dengan durasi > 240 menit:")
    cols = [
        "entry_time", "exit_time", "direction",
        "exit_reason", "R", "elapsed_minutes"
    ]
    print(
        df.loc[df["elapsed_minutes"] > 240, cols]
        .sort_values("elapsed_minutes", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    print("\nDurasi menurut alasan keluar:")
    print(
        df.groupby("exit_reason")["elapsed_minutes"]
        .agg(["count", "median", "max"])
        .round(2)
        .to_string()
    )

print("\nAudit durasi selesai.")
