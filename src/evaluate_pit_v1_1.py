from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "backtest_pit_v1.csv"
OUT = ROOT / "data" / "processed"

df = pd.read_csv(INPUT)
df.columns = df.columns.str.strip()

required = [
    "macro_score",
    "real_yield_age_days",
    "forward_return_1bar_pct",
    "forward_return_3bar_pct",
    "forward_return_5bar_pct",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Kolom wajib tidak ditemukan: {missing}\n"
        f"Kolom tersedia: {df.columns.tolist()}"
    )

# Urutkan semua candle sebelum membuat indeks dan menyaring sinyal.
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
elif "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
else:
    raise ValueError("Kolom time/date tidak ditemukan.")

df["original_bar_index"] = np.arange(len(df), dtype=np.int64)
df["macro_score"] = pd.to_numeric(df["macro_score"], errors="coerce")
df["real_yield_age_days"] = pd.to_numeric(
    df["real_yield_age_days"], errors="coerce"
)
df["year"] = df["time"].dt.year

base = df[
    df["macro_score"].notna()
    & (df["macro_score"] != 0)
    & df["real_yield_age_days"].between(0, 3)
].copy()

base["direction"] = np.sign(base["macro_score"])

def evaluate(sample, horizon):
    ret_col = f"forward_return_{horizon}bar_pct"
    x = sample.copy()
    x[ret_col] = pd.to_numeric(x[ret_col], errors="coerce")
    x = x.dropna(subset=[ret_col, "direction"])

    columns = {
        "n_signals": 0,
        "directional_accuracy_pct": np.nan,
        "strategy_mean_return_pct": np.nan,
        "strategy_median_return_pct": np.nan,
        "strategy_win_rate_pct": np.nan,
        "always_long_mean_return_pct": np.nan,
        "mean_edge_vs_always_long_pct": np.nan,
        "edge_ci95_low_pct": np.nan,
        "edge_ci95_high_pct": np.nan,
    }
    if x.empty:
        return columns

    x["strategy_return_pct"] = x["direction"] * x[ret_col]
    x["edge_pct"] = x["strategy_return_pct"] - x[ret_col]

    # Bootstrap CI 95% untuk rata-rata selisih strategi vs always long.
    values = x["edge_pct"].to_numpy(dtype=float)
    rng = np.random.default_rng(20261021)
    boot = np.empty(3000)

    for i in range(len(boot)):
        boot[i] = rng.choice(values, size=len(values), replace=True).mean()

    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    correct_col = f"direction_correct_{horizon}bar"
    if correct_col in x.columns:
        accuracy = (
            pd.to_numeric(x[correct_col], errors="coerce").mean() * 100
        )
    else:
        accuracy = (x["strategy_return_pct"] > 0).mean() * 100

    return {
        "n_signals": len(x),
        "directional_accuracy_pct": accuracy,
        "strategy_mean_return_pct": x["strategy_return_pct"].mean(),
        "strategy_median_return_pct": x["strategy_return_pct"].median(),
        "strategy_win_rate_pct": (
            x["strategy_return_pct"] > 0
        ).mean() * 100,
        "always_long_mean_return_pct": x[ret_col].mean(),
        "mean_edge_vs_always_long_pct": x["edge_pct"].mean(),
        "edge_ci95_low_pct": ci_low,
        "edge_ci95_high_pct": ci_high,
    }

summary = []
annual = []
nonoverlap = []

for h in (1, 3, 5):
    ret_col = f"forward_return_{h}bar_pct"
    sample = base.dropna(subset=[ret_col]).copy()

    result = {"horizon_bars": h}
    result.update(evaluate(sample, h))
    summary.append(result)

    for year, group in sample.groupby("year", dropna=True):
        result = {"year": int(year), "horizon_bars": h}
        result.update(evaluate(group, h))
        annual.append(result)

    # Kelompok offset dari indeks candle penuh.
    for offset in range(h):
        part = sample[
            sample["original_bar_index"] % h == offset
        ].copy()

        result = {"horizon_bars": h, "offset": offset}
        result.update(evaluate(part, h))
        nonoverlap.append(result)

summary_df = pd.DataFrame(summary)
annual_df = pd.DataFrame(annual)
nonoverlap_df = pd.DataFrame(nonoverlap)

paths = {
    "summary": OUT / "pit_v1_1_evaluation_summary.csv",
    "annual": OUT / "pit_v1_1_evaluation_annual.csv",
    "nonoverlap": OUT / "pit_v1_1_nonoverlap.csv",
}
summary_df.to_csv(paths["summary"], index=False)
annual_df.to_csv(paths["annual"], index=False)
nonoverlap_df.to_csv(paths["nonoverlap"], index=False)

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 20)

print("\n=== RINGKASAN STRATEGI VS ALWAYS LONG ===")
print(summary_df.round(4).to_string(index=False))

print("\n=== HASIL PER TAHUN ===")
print(annual_df.round(4).to_string(index=False))

print("\n=== NON-OVERLAP BERDASARKAN INDEKS CANDLE ASLI ===")
print(nonoverlap_df.round(4).to_string(index=False))

print("\nFile tersimpan:")
for path in paths.values():
    print(path)

print("\nCatatan:")
print("- Edge positif berarti strategi mengungguli always long pada sampel.")
print("- CI 95% yang melintasi nol belum mendukung edge yang meyakinkan.")
print("- Bootstrap ini belum mengatasi dependensi serial.")
print("- Return belum dikurangi spread, komisi, dan slippage.")
print("- Indeks dibuat dari urutan candle yang tersedia; celah kalender tetap ada.")
