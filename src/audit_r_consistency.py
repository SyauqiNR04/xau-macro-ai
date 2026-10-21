import pandas as pd
import numpy as np

FILES = {
    "Technical only": "reports/technical_only_trades.csv",
    "Technical + macro": "reports/technical_macro_trades.csv",
}

for name, path in FILES.items():
    df = pd.read_csv(path)

    for c in ["entry_price", "exit_price", "stop", "pnl_price", "R"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    risk = (df["entry_price"] - df["stop"]).abs()
    df["risk_price"] = risk
    df["R_recomputed"] = np.where(
        risk > 0,
        df["pnl_price"] / risk,
        np.nan
    )
    df["R_diff"] = df["R"] - df["R_recomputed"]

    print("\n" + "=" * 70)
    print(name)
    print("Jumlah transaksi:", len(df))
    print("Risiko awal tidak valid:", int((risk <= 0).sum()))
    print("Nilai R tidak tersedia:", int(df["R"].isna().sum()))

    valid = df.dropna(subset=["R", "R_recomputed"])
    print("Selisih absolut R maksimum:",
          round(valid["R_diff"].abs().max(), 6))
    print("Transaksi dengan selisih R > 0.02:",
          int((valid["R_diff"].abs() > 0.02).sum()))

    print("\nPerbandingan berdasarkan exit:")
    print(
        df.groupby("exit_reason").agg(
            trades=("R", "count"),
            mean_R=("R", "mean"),
            mean_recomputed_R=("R_recomputed", "mean"),
            max_abs_diff=("R_diff", lambda x: x.abs().max())
        ).round(4).to_string()
    )

    print("\nContoh 10 selisih terbesar:")
    cols = [
        "entry_time", "direction", "exit_reason",
        "entry_price", "exit_price", "stop",
        "pnl_price", "R", "R_recomputed", "R_diff"
    ]
    print(
        valid.loc[valid["R_diff"].abs().nlargest(10).index, cols]
        .to_string(index=False)
    )
