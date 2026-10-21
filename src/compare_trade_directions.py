from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SRC = ROOT / "src" / "backtest_intraday_v1.py"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("backtest_v1", SRC)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

print("Membangun dataset menggunakan pipeline asli...")
df = bt.build_dataset()
print("Jumlah candle:", len(df))

def metrics(trades):
    if trades.empty:
        return {
            "trades": 0, "win_rate_pct": np.nan,
            "expectancy_R": np.nan, "profit_factor": np.nan,
            "net_R": 0.0, "max_drawdown_R": 0.0
        }

    r = pd.to_numeric(trades["R"], errors="coerce").dropna()
    if r.empty:
        return {
            "trades": 0, "win_rate_pct": np.nan,
            "expectancy_R": np.nan, "profit_factor": np.nan,
            "net_R": 0.0, "max_drawdown_R": 0.0
        }

    gross_win = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.nan
    equity = r.cumsum()
    drawdown = equity - equity.cummax()

    return {
        "trades": len(r),
        "win_rate_pct": round((r > 0).mean() * 100, 4),
        "expectancy_R": round(r.mean(), 5),
        "profit_factor": round(pf, 5) if np.isfinite(pf) else np.nan,
        "net_R": round(r.sum(), 4),
        "max_drawdown_R": round(drawdown.min(), 4)
    }

variants = [
    ("Technical | LONG+SHORT", False, "ALL"),
    ("Technical | LONG only", False, "LONG"),
    ("Technical | SHORT only", False, "SHORT"),
    ("Macro | LONG+SHORT", True, "ALL"),
    ("Macro | LONG only", True, "LONG"),
    ("Macro | SHORT only", True, "SHORT"),
]

all_trades = []
summary_rows = []
oos_start = pd.Timestamp("2024-01-01", tz="UTC")

for name, use_macro, direction in variants:
    test_df = df.copy()

    # Filter sinyal SEBELUM simulasi dimulai, bukan setelah trade selesai.
    if direction == "LONG":
        test_df.loc[test_df["entry_signal"] != 1, "entry_signal"] = 0
    elif direction == "SHORT":
        test_df.loc[test_df["entry_signal"] != -1, "entry_signal"] = 0

    trades = bt.simulate(test_df, use_macro=use_macro)

    if not trades.empty:
        trades = trades.copy()
        trades["strategy"] = name
        trades["entry_time"] = pd.to_datetime(
            trades["entry_time"], utc=True
        )
        trades["year"] = trades["entry_time"].dt.year
        all_trades.append(trades)

    for period, subset in [
        ("FULL", trades),
        ("OOS_2024+", trades[
            trades["entry_time"] >= oos_start
        ] if not trades.empty else trades)
    ]:
        row = {
            "strategy": name,
            "period": period,
            **metrics(subset)
        }
        summary_rows.append(row)

if all_trades:
    combined = pd.concat(all_trades, ignore_index=True)
    combined.to_csv(OUT / "direction_comparison_trades.csv", index=False)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "direction_comparison_summary.csv", index=False)

print("\n=== PERBANDINGAN FULL DAN OOS ===")
print(summary.to_string(index=False))

if all_trades:
    combined = pd.concat(all_trades, ignore_index=True)
    yearly_rows = []

    for (strategy, year), group in combined.groupby(["strategy", "year"]):
        yearly_rows.append({
            "strategy": strategy,
            "year": year,
            **metrics(group)
        })

    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(OUT / "direction_comparison_yearly.csv", index=False)

    print("\n=== HASIL TAHUNAN ===")
    print(yearly.to_string(index=False))

print("\nFile tersimpan:")
print(OUT / "direction_comparison_summary.csv")
print(OUT / "direction_comparison_yearly.csv")
print(OUT / "direction_comparison_trades.csv")
print("\nCatatan: hasil tetap bergantung pada asumsi spread, slippage, dan eksekusi simulator.")
