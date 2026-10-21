
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "backtest_dataset.csv"
OUTPUT = ROOT / "data" / "processed" / "macro_backtest_evaluation.csv"

HORIZONS = [1, 3, 5]
TEST_START = pd.Timestamp("2023-01-01")


def classify_score(score):
    if pd.isna(score):
        return "Tidak tersedia"
    if score >= 50:
        return "Bullish"
    if score <= -50:
        return "Bearish"
    return "Netral"


def summarize(group, horizon):
    col = f"forward_return_{horizon}d"

    valid = group.loc[
        group["macro_score_v2"].notna()
        & group[col].notna()
    ]

    if valid.empty:
        return {
            "n": 0,
            "mean_return_pct": None,
            "median_return_pct": None,
            "positive_return_pct": None,
            "score_return_spearman": None,
        }

    return {
        "n": len(valid),
        "mean_return_pct": valid[col].mean() * 100,
        "median_return_pct": valid[col].median() * 100,
        "positive_return_pct": (valid[col] > 0).mean() * 100,
        "score_return_spearman": valid[
            ["macro_score_v2", col]
        ].corr(method="spearman").iloc[0, 1],
    }


def main():
    df = pd.read_csv(INPUT, parse_dates=["gold_date"])

    required = {
        "gold_date",
        "macro_score_v2",
        "macro_age_days",
        "forward_return_1d",
        "forward_return_3d",
        "forward_return_5d",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {sorted(missing)}")

    # Semua baris harus sudah memakai skor dari tanggal sebelumnya.
    # Batasi evaluasi awal pada skor yang tidak stale.
    eligible = df.loc[
        df["macro_score_v2"].notna()
        & df["macro_age_days"].between(0, 3)
    ].copy()

    eligible["regime"] = eligible["macro_score_v2"].apply(
        classify_score
    )

    eligible["period"] = (
        eligible["gold_date"].ge(TEST_START)
        .map({True: "Test_2023_plus", False: "Earlier"})
    )

    results = []

    # Evaluasi keseluruhan dan evaluasi per periode.
    for period_name, period_df in [
        ("All", eligible),
        ("Earlier", eligible[eligible["period"] == "Earlier"]),
        (
            "Test_2023_plus",
            eligible[eligible["period"] == "Test_2023_plus"],
        ),
    ]:
        for horizon in HORIZONS:
            for regime in ["Bullish", "Netral", "Bearish"]:
                group = period_df[period_df["regime"] == regime]
                metrics = summarize(group, horizon)

                results.append({
                    "period": period_name,
                    "regime": regime,
                    "horizon_candles": horizon,
                    **metrics,
                })

            # Korelasi dihitung pada seluruh regime di periode tersebut.
            metrics = summarize(period_df, horizon)
            results.append({
                "period": period_name,
                "regime": "Semua regime",
                "horizon_candles": horizon,
                **metrics,
            })

    report = pd.DataFrame(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT, index=False)

    print("=" * 75)
    print("MACRO SCORE V2 - DESCRIPTIVE BACKTEST")
    print("=" * 75)
    print(f"Dataset rows           : {len(df):,}")
    print(f"Eligible rows          : {len(eligible):,}")
    print(f"Excluded rows          : {len(df) - len(eligible):,}")
    print(f"Test period starts     : {TEST_START.date()}")
    print(f"Output                 : {OUTPUT}")

    print("\nSignal distribution:")
    print(eligible["regime"].value_counts().to_string())

    cols = [
        "period",
        "regime",
        "horizon_candles",
        "n",
        "mean_return_pct",
        "median_return_pct",
        "positive_return_pct",
        "score_return_spearman",
    ]
    print("\nPerformance summary:")
    print(
        report[cols].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
