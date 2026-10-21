import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("reports/cooldown_comparison_trades.csv")
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df["R"] = pd.to_numeric(df["R"], errors="coerce")
df = df.sort_values("entry_time")
df["year"] = df["entry_time"].dt.year

def calc(g):
    r = g["R"].dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return pd.Series({
            "trades": 0, "net_R": 0.0,
            "expectancy_R": np.nan, "profit_factor": np.nan,
            "max_drawdown_R": np.nan
        })

    equity = np.concatenate(([0.0], np.cumsum(r)))
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()

    return pd.Series({
        "trades": len(r),
        "net_R": r.sum(),
        "expectancy_R": r.mean(),
        "profit_factor": gains / losses if losses > 0 else np.nan,
        "max_drawdown_R": drawdown.min()
    })

annual = (
    df.groupby(["model", "scenario", "year"])
      .apply(calc, include_groups=False)
      .reset_index()
      .sort_values(["model", "scenario", "year"])
)
annual.to_csv("reports/cooldown_annual_audit_corrected.csv", index=False)

print("=== DRAWDOWN TAHUNAN YANG SUDAH DIPERBAIKI ===")
print(annual.round(4).to_string(index=False))

oos = df[df["entry_time"] >= pd.Timestamp("2024-01-01", tz="UTC")]
summary = (
    oos.groupby(["model", "scenario"])
       .apply(calc, include_groups=False)
       .reset_index()
       .sort_values(["model", "scenario"])
)
summary.to_csv("reports/cooldown_oos_audit_corrected.csv", index=False)

print("\n=== DRAWDOWN OOS KUMULATIF ===")
print(summary.round(4).to_string(index=False))
