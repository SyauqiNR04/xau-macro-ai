
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "backtest_dataset.csv"
OUTPUT = ROOT / "data" / "processed" / "annual_stability.csv"

TEST_START = pd.Timestamp("2023-01-01")
HORIZONS = [1, 3, 5]


def main():
    df = pd.read_csv(INPUT, parse_dates=["gold_date"])

    df = df.loc[
        df["gold_date"].ge(TEST_START)
        & df["macro_score_v2"].notna()
        & df["macro_age_days"].between(0, 3)
    ].copy()

    # Hanya sinyal dengan arah yang cukup tegas.
    df["prediction"] = 0
    df.loc[df["macro_score_v2"] >= 50, "prediction"] = 1
    df.loc[df["macro_score_v2"] <= -50, "prediction"] = -1

    df["year"] = df["gold_date"].dt.year
    rows = []

    for year, yearly in df.groupby("year"):
        for horizon in HORIZONS:
            col = f"forward_return_{horizon}d"
            valid = yearly.dropna(subset=[col])

            if valid.empty:
                continue

            active = valid[valid["prediction"] != 0]

            if len(active):
                correct = (
                    active["prediction"]
                    * active[col].apply(lambda x: 1 if x > 0 else -1)
                ) > 0
                accuracy = correct.mean() * 100
            else:
                accuracy = float("nan")

            rows.append({
                "year": year,
                "horizon": horizon,
                "n_all": len(valid),
                "baseline_positive_pct":
                    (valid[col] > 0).mean() * 100,
                "n_directional": len(active),
                "directional_accuracy_pct": accuracy,
                "bullish_mean_return_pct":
                    valid.loc[
                        valid["prediction"] == 1, col
                    ].mean() * 100,
                "bearish_mean_return_pct":
                    valid.loc[
                        valid["prediction"] == -1, col
                    ].mean() * 100,
            })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)

    print("=== ANNUAL STABILITY: TEST 2023+ ===")
    print(result.round(3).to_string(index=False))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
