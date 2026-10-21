import pandas as pd
from pathlib import Path

files = {
    "Technical only": "reports/technical_only_trades.csv",
    "Technical + macro": "reports/technical_macro_trades.csv",
}

for name, path in files.items():
    df = pd.read_csv(path)
    print(f"\n{'=' * 65}")
    print(name)
    print("Kolom:", df.columns.tolist())

    # Temukan kolom tanggal transaksi
    date_col = next(
        (c for c in ["entry_time", "entry_date", "time", "entry_timestamp"]
         if c in df.columns),
        None
    )
    if date_col is None:
        print("Kolom waktu masuk tidak ditemukan; periksa daftar kolom di atas.")
        continue

    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    oos = df[df[date_col] >= "2024-01-01"].copy()

    print("\nJumlah transaksi OOS:", len(oos))

    # Ringkasan per tahun jika tersedia kolom hasil R
    r_col = next(
        (c for c in ["R", "result_R", "net_R", "pnl_R", "return_R"]
         if c in oos.columns),
        None
    )
    if r_col:
        oos[r_col] = pd.to_numeric(oos[r_col], errors="coerce")
        oos["year"] = oos[date_col].dt.year
        print("\nHasil per tahun:")
        print(oos.groupby("year")[r_col].agg(
            trades="count", expectancy="mean", net_R="sum"
        ).round(4))

    # Audit jenis exit
    exit_col = next(
        (c for c in ["exit_reason", "reason", "exit_type"]
         if c in oos.columns),
        None
    )
    if exit_col:
        print("\nJenis exit OOS:")
        print(oos[exit_col].value_counts(dropna=False))

    # Audit arah transaksi
    side_col = next(
        (c for c in ["direction", "side", "signal"]
         if c in oos.columns),
        None
    )
    if side_col:
        print("\nArah transaksi OOS:")
        print(oos[side_col].value_counts(dropna=False))
