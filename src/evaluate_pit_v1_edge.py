from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SOURCE = ROOT / "data" / "processed" / "backtest_pit_v1.csv"
OUT_SUMMARY = ROOT / "data" / "processed" / "pit_v1_evaluation_summary.csv"
OUT_ANNUAL = ROOT / "data" / "processed" / "pit_v1_evaluation_annual.csv"
OUT_NONOVERLAP = ROOT / "data" / "processed" / "pit_v1_nonoverlap.csv"

if not SOURCE.exists():
    raise SystemExit(f"File tidak ditemukan: {SOURCE}")

df = pd.read_csv(SOURCE, parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["macro_score"] = pd.to_numeric(df["macro_score"], errors="coerce")
df["real_yield_age_days"] = pd.to_numeric(
    df["real_yield_age_days"], errors="coerce"
)

df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
df["year"] = df["date"].dt.year

eligible = (
    df["macro_score"].notna()
    & df["macro_score"].ne(0)
    & df["real_yield_age_days"].between(0, 3)
)

summaries = []
annual_rows = []
nonoverlap_rows = []

for horizon in [1, 3, 5]:
    ret_col = f"forward_return_{horizon}bar_pct"
    if ret_col not in df.columns:
        continue

    ret = pd.to_numeric(df[ret_col], errors="coerce")
    valid = eligible & ret.notna()
    sample = df.loc[valid].copy()
    sample["forward_return_pct"] = ret.loc[valid]
    sample["signal_direction"] = np.sign(sample["macro_score"])

    # Return strategi arah: long saat bullish, short saat bearish.
    sample["strategy_return_pct"] = (
        sample["signal_direction"] * sample["forward_return_pct"]
    )

    # Baseline selalu long pada periode dan sampel yang sama.
    strategy = sample["strategy_return_pct"]
    buy_hold = sample["forward_return_pct"]

    accuracy = (
        np.sign(sample["macro_score"])
        == np.sign(sample["forward_return_pct"])
    ).mean() * 100

    summaries.append({
        "horizon_bars": horizon,
        "n_signals": len(sample),
        "directional_accuracy_pct": accuracy,
        "strategy_mean_return_pct": strategy.mean(),
        "strategy_median_return_pct": strategy.median(),
        "strategy_win_rate_pct": (strategy > 0).mean() * 100,
        "always_long_mean_return_pct": buy_hold.mean(),
        "mean_edge_vs_always_long_pct": strategy.mean() - buy_hold.mean(),
        "strategy_positive_fraction": (strategy > 0).mean(),
    })

    for year, group in sample.groupby("year"):
        annual_rows.append({
            "year": int(year),
            "horizon_bars": horizon,
            "n_signals": len(group),
            "directional_accuracy_pct": (
                np.sign(group["macro_score"])
                == np.sign(group["forward_return_pct"])
            ).mean() * 100,
            "strategy_mean_return_pct": group["strategy_return_pct"].mean(),
            "always_long_mean_return_pct": group["forward_return_pct"].mean(),
            "mean_edge_vs_always_long_pct": (
                group["strategy_return_pct"].mean()
                - group["forward_return_pct"].mean()
            ),
        })

    # Uji sensitivitas: ambil satu sinyal setiap N candle.
    # Ini mengurangi, tetapi tidak menghilangkan, ketergantungan sampel.
    for offset in range(horizon):
        sub = sample.iloc[offset::horizon].copy()
        if sub.empty:
            continue
        nonoverlap_rows.append({
            "horizon_bars": horizon,
            "offset": offset,
            "n_signals": len(sub),
            "directional_accuracy_pct": (
                np.sign(sub["macro_score"])
                == np.sign(sub["forward_return_pct"])
            ).mean() * 100,
            "strategy_mean_return_pct": (
                sub["signal_direction"] * sub["forward_return_pct"]
            ).mean(),
            "always_long_mean_return_pct": sub["forward_return_pct"].mean(),
        })

summary = pd.DataFrame(summaries)
annual = pd.DataFrame(annual_rows)
nonoverlap = pd.DataFrame(nonoverlap_rows)

summary.to_csv(OUT_SUMMARY, index=False)
annual.to_csv(OUT_ANNUAL, index=False)
nonoverlap.to_csv(OUT_NONOVERLAP, index=False)

print("=== RINGKASAN STRATEGI VS ALWAYS LONG ===")
print(summary.round(4).to_string(index=False))

print("\n=== HASIL PER TAHUN ===")
if not annual.empty:
    print(annual.round(4).to_string(index=False))

print("\n=== UJI NON-OVERLAP ===")
if not nonoverlap.empty:
    print(nonoverlap.round(4).to_string(index=False))

print("\nFile tersimpan:")
print(OUT_SUMMARY)
print(OUT_ANNUAL)
print(OUT_NONOVERLAP)

print("\nPeringatan:")
print("- Return masih gross, belum dikurangi spread, komisi, dan slippage.")
print("- Sampel historis ini bukan jaminan kinerja masa depan.")
print("- Uji non-overlap mengurangi overlap, tetapi tidak menjamin independensi.")
print("- Belum ada confidence interval atau pengujian signifikansi statistik.")
