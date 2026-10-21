import pandas as pd
import numpy as np
from pathlib import Path

src = Path("reports/cooldown_comparison_trades.csv")
out = Path("reports")
df = pd.read_csv(src)

df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df["R"] = pd.to_numeric(df["R"], errors="coerce")
df["year"] = df["entry_time"].dt.year

def metrics(g):
    r = g["R"].dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    equity = pd.concat([pd.Series([0.0]), r.reset_index(drop=True)])
    dd = equity - equity.cummax()

    return pd.Series({
        "trades": len(r),
        "win_rate_pct": (r > 0).mean() * 100 if len(r) else np.nan,
        "expectancy_R": r.mean() if len(r) else np.nan,
        "profit_factor": gains / losses if losses > 0 else np.nan,
        "net_R": r.sum(),
        "max_drawdown_R": dd.min() if len(r) else np.nan,
    })

keys = ["model", "scenario", "year"]
annual = (
    df.groupby(keys, dropna=False)
      .apply(metrics, include_groups=False)
      .reset_index()
      .sort_values(keys)
)

annual.to_csv(out / "cooldown_annual_audit.csv", index=False)

print("=== HASIL TAHUNAN ===")
print(annual.round(4).to_string(index=False))

print("\n=== KHUSUS OOS 2024+ ===")
oos = annual[annual["year"] >= 2024]
print(oos.round(4).to_string(index=False))

print("\n=== RINGKASAN OOS PER SKENARIO ===")
summary = (
    df[df["entry_time"] >= pd.Timestamp("2024-01-01", tz="UTC")]
    .groupby(["model", "scenario"], dropna=False)
    .apply(metrics, include_groups=False)
    .reset_index()
    .sort_values(["model", "scenario"])
)
print(summary.round(4).to_string(index=False))

print("\nFile disimpan: reports/cooldown_annual_audit.csv")
