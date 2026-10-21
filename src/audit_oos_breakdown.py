import pandas as pd

FILES = {
    "Technical only": "reports/technical_only_trades.csv",
    "Technical + macro": "reports/technical_macro_trades.csv",
}

for name, path in FILES.items():
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["R"] = pd.to_numeric(df["R"], errors="coerce")
    df = df[df["entry_time"] >= "2024-01-01"].copy()

    df["year"] = df["entry_time"].dt.year
    df["month"] = df["entry_time"].dt.to_period("M").astype(str)

    print("\n" + "=" * 72)
    print(name)

    print("\n--- HASIL BERDASARKAN TAHUN DAN ARAH ---")
    print(
        df.groupby(["year", "direction"])["R"]
        .agg(trades="count", expectancy="mean", net_R="sum")
        .round(4)
        .to_string()
    )

    print("\n--- HASIL BERDASARKAN TAHUN DAN JENIS EXIT ---")
    print(
        df.groupby(["year", "exit_reason"])["R"]
        .agg(trades="count", expectancy="mean", net_R="sum")
        .round(4)
        .to_string()
    )

    print("\n--- HASIL BULANAN ---")
    monthly = df.groupby("month")["R"].agg(
        trades="count", expectancy="mean", net_R="sum"
    )
    print(monthly.round(4).to_string())

    print("\n--- RINGKASAN KINERJA ---")
    print("Total transaksi:", len(df))
    print("Win rate (%):", round((df["R"] > 0).mean() * 100, 2))
    print("Expectancy (R):", round(df["R"].mean(), 4))
    print("Net R:", round(df["R"].sum(), 4))
