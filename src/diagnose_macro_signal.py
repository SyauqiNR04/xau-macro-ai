
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "backtest_dataset.csv"
OUT = ROOT / "data" / "processed"

HORIZONS = [1, 3, 5]
TEST_START = pd.Timestamp("2023-01-01")


def regime(score):
    if score >= 50:
        return "Bullish"
    if score <= -50:
        return "Bearish"
    return "Netral"


def metrics(group, horizon):
    col = f"forward_return_{horizon}d"
    r = group[col].dropna()

    if r.empty:
        return {
            "n": 0,
            "mean_pct": None,
            "median_pct": None,
            "win_rate_pct": None,
        }

    return {
        "n": len(r),
        "mean_pct": r.mean() * 100,
        "median_pct": r.median() * 100,
        "win_rate_pct": (r > 0).mean() * 100,
    }


def main():
    df = pd.read_csv(INPUT, parse_dates=["gold_date"])

    df = df.loc[
        df["macro_score_v2"].notna()
        & df["macro_age_days"].between(0, 3)
    ].copy()

    df["regime"] = df["macro_score_v2"].apply(regime)
    df["year"] = df["gold_date"].dt.year

    # Evaluasi periode test yang sama dengan analisis sebelumnya.
    test = df[df["gold_date"] >= TEST_START].copy()

    rows = []
    for horizon in HORIZONS:
        col = f"forward_return_{horizon}d"

        baseline = metrics(test, horizon)
        rows.append({
            "period": "Test_2023_plus",
            "year": "All",
            "regime": "Baseline",
            "horizon": horizon,
            **baseline,
        })

        for name in ["Bullish", "Netral", "Bearish"]:
            group = test[test["regime"] == name]
            rows.append({
                "period": "Test_2023_plus",
                "year": "All",
                "regime": name,
                "horizon": horizon,
                **metrics(group, horizon),
            })

        # Directional accuracy: netral tidak dihitung sebagai prediksi.
        directional = test[
            (test["macro_score_v2"] > 0)
            | (test["macro_score_v2"] < 0)
        ].dropna(subset=[col]).copy()

        correct = (
            (directional["macro_score_v2"] > 0)
            & (directional[col] > 0)
        ) | (
            (directional["macro_score_v2"] < 0)
            & (directional[col] < 0)
        )

        print(
            f"Horizon {horizon}: "
            f"directional accuracy = {correct.mean() * 100:.2f}% "
            f"(n={len(directional)})"
        )

        # Breakdown tahunan untuk memeriksa kestabilan.
        for year, year_df in test.groupby("year"):
            for name in ["Bullish", "Netral", "Bearish"]:
                group = year_df[year_df["regime"] == name]
                rows.append({
                    "period": "Test_2023_plus",
                    "year": int(year),
                    "regime": name,
                    "horizon": horizon,
                    **metrics(group, horizon),
                })

    report = pd.DataFrame(rows)
    path = OUT / "macro_signal_diagnostics.csv"
    report.to_csv(path, index=False)

    print("\n=== BASELINE VS REGIME: TEST 2023+ ===")
    print(
        report[
            (report["year"] == "All")
            & (report["regime"].isin(
                ["Baseline", "Bullish", "Netral", "Bearish"]
            ))
        ].round(4).to_string(index=False)
    )

    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
