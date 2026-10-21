
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

GOLD_PATH = ROOT / "data/raw/xauusd_d1.csv"
MACRO_PATH = ROOT / "data/processed/macro_score_v2.csv"
OUT_DIR = ROOT / "data/processed"

HORIZONS = [1, 3, 5]
MAX_SIGNAL_AGE_DAYS = 4


def load_data():
    gold = pd.read_csv(GOLD_PATH)
    macro = pd.read_csv(MACRO_PATH)

    required_gold = {"time", "open", "high", "low", "close"}
    required_macro = {
        "observation_date",
        "macro_score_v2",
        "macro_bias_v2",
    }

    if not required_gold.issubset(gold.columns):
        raise ValueError(
            f"Kolom gold tidak lengkap: {required_gold - set(gold.columns)}"
        )

    if not required_macro.issubset(macro.columns):
        raise ValueError(
            f"Kolom macro tidak lengkap: {required_macro - set(macro.columns)}"
        )

    gold["date"] = pd.to_datetime(gold["time"], errors="coerce").dt.normalize()
    macro["date"] = pd.to_datetime(
        macro["observation_date"], errors="coerce"
    ).dt.normalize()

    for col in ["open", "high", "low", "close"]:
        gold[col] = pd.to_numeric(gold[col], errors="coerce")

    macro["macro_score_v2"] = pd.to_numeric(
        macro["macro_score_v2"], errors="coerce"
    )

    gold = gold.dropna(subset=["date", "open", "high", "low", "close"])
    macro = macro.dropna(subset=["date", "macro_score_v2"])

    if gold["date"].duplicated().any():
        raise ValueError("Tanggal gold duplikat ditemukan.")

    if macro["date"].duplicated().any():
        raise ValueError("Tanggal macro duplikat ditemukan.")

    if (gold[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Ada harga OHLC nol atau negatif.")

    if (gold["high"] < gold["low"]).any():
        raise ValueError("Ditemukan high yang lebih rendah daripada low.")

    gold = gold.sort_values("date").reset_index(drop=True)
    macro = macro.sort_values("date").reset_index(drop=True)

    return gold, macro


def align_signals(gold, macro):
    # Sinyal hanya boleh berasal dari tanggal yang STRICTLY lebih awal
    # daripada tanggal candle entry.
    aligned = pd.merge_asof(
        gold.sort_values("date"),
        macro.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_macro"),
    )

    aligned["signal_age_days"] = (
        aligned["date"] - aligned["observation_date"].pipe(
            lambda s: pd.to_datetime(s, errors="coerce")
        ).dt.normalize()
    ).dt.days

    # Jangan memakai sinyal yang terlalu lama.
    stale = (
        aligned["signal_age_days"].isna()
        | (aligned["signal_age_days"] < 1)
        | (aligned["signal_age_days"] > MAX_SIGNAL_AGE_DAYS)
    )

    aligned.loc[stale, "macro_score_v2"] = np.nan
    aligned.loc[stale, "macro_bias_v2"] = "Tidak tersedia"

    return aligned


def calculate_returns(data):
    # Entry pada OPEN candle pertama setelah tanggal sinyal makro.
    # Horizon 1: entry open hari ini, exit close hari ini.
    # Horizon 3: entry open hari ini, exit close dua candle kemudian.
    # Horizon 5: entry open hari ini, exit close empat candle kemudian.
    for h in HORIZONS:
        exit_close = data["close"].shift(-(h - 1))
        price_return = (exit_close / data["open"] - 1) * 100

        data[f"price_return_{h}d_pct"] = price_return

        score = data["macro_score_v2"]
        bias = data["macro_bias_v2"].astype(str).str.lower()

        # Bias positif = bullish emas; bias negatif = bearish emas.
        # Netral tidak dianggap sebagai posisi trading.
        direction = pd.Series(np.nan, index=data.index, dtype=float)
        direction.loc[score > 0] = 1.0
        direction.loc[score < 0] = -1.0

        # Gunakan bias untuk memastikan sinyal konsisten dengan skor.
        direction.loc[bias.str.contains("bullish", na=False)] = 1.0
        direction.loc[bias.str.contains("bearish", na=False)] = -1.0
        direction.loc[
            bias.str.contains("netral|neutral|tidak tersedia", na=False)
        ] = np.nan

        data[f"strategy_return_{h}d_pct"] = (
            price_return * direction
        )

        # Akurasi arah: hanya sinyal bullish/bearish dan return nonnol.
        correct = (
            ((direction == 1) & (price_return > 0))
            | ((direction == -1) & (price_return < 0))
        )
        eligible = direction.notna() & price_return.notna() & (price_return != 0)

        data[f"direction_correct_{h}d"] = np.where(
            eligible, correct.astype(float), np.nan
        )

    return data


def summarize(data):
    summary_rows = []

    for h in HORIZONS:
        price_col = f"price_return_{h}d_pct"
        strat_col = f"strategy_return_{h}d_pct"

        groups = [
            ("Cenderung bullish emas", data["macro_bias_v2"].astype(str).str.lower().str.contains("bullish", na=False)),
            ("Cenderung bearish emas", data["macro_bias_v2"].astype(str).str.lower().str.contains("bearish", na=False)),
            ("Netral", data["macro_bias_v2"].astype(str).str.lower().str.contains("netral|neutral", na=False)),
            ("Baseline semua sinyal valid", data["macro_score_v2"].notna()),
        ]

        for label, mask in groups:
            subset = data.loc[mask & data[price_col].notna()]
            strategy_returns = subset[strat_col].dropna()

            summary_rows.append({
                "group": label,
                "horizon_candles": h,
                "price_return_samples": len(subset),
                "mean_price_return_pct": subset[price_col].mean(),
                "median_price_return_pct": subset[price_col].median(),
                "price_win_rate_pct": (
                    (subset[price_col] > 0).mean() * 100
                    if len(subset) else np.nan
                ),
                "strategy_samples": len(strategy_returns),
                "mean_strategy_return_pct": strategy_returns.mean(),
                "median_strategy_return_pct": strategy_returns.median(),
                "strategy_win_rate_pct": (
                    (strategy_returns > 0).mean() * 100
                    if len(strategy_returns) else np.nan
                ),
            })

    summary = pd.DataFrame(summary_rows)

    accuracy_rows = []
    for h in HORIZONS:
        col = f"direction_correct_{h}d"
        values = data[col].dropna()

        accuracy_rows.append({
            "horizon_candles": h,
            "directional_samples": len(values),
            "directional_accuracy_pct": (
                values.mean() * 100 if len(values) else np.nan
            ),
        })

    accuracy = pd.DataFrame(accuracy_rows)
    return summary, accuracy


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold, macro = load_data()
    aligned = align_signals(gold, macro)
    result = calculate_returns(aligned)
    summary, accuracy = summarize(result)

    detail_path = OUT_DIR / "backtest_macro_gold_detail_v2.csv"
    summary_path = OUT_DIR / "backtest_macro_gold_summary_v2.csv"
    accuracy_path = OUT_DIR / "backtest_macro_gold_accuracy_v2.csv"
    audit_path = OUT_DIR / "backtest_macro_gold_audit_v2.txt"

    result.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    accuracy.to_csv(accuracy_path, index=False)

    has_signal = result["macro_score_v2"].notna()
    audit = [
        "=== AUDIT BACKTEST V2 ===",
        f"gold_rows: {len(gold)}",
        f"macro_rows_with_score: {len(macro)}",
        f"gold_rows_with_usable_signal: {int(has_signal.sum())}",
        f"gold_rows_without_usable_signal: {int((~has_signal).sum())}",
        f"gold_first_date: {gold['date'].min().date()}",
        f"gold_last_date: {gold['date'].max().date()}",
        f"macro_first_date: {macro['date'].min().date()}",
        f"macro_last_date: {macro['date'].max().date()}",
        f"max_signal_age_days: {MAX_SIGNAL_AGE_DAYS}",
        "signal_alignment: macro date must be strictly earlier than entry candle date",
        "entry: open of the aligned gold candle",
        "exit: close after H-1 additional candle observations",
        "costs_included: False",
        "feed: Dukascopy XAU-USD BID; may differ from Exness execution feed",
        "warning: results are historical diagnostics, not proof of future profitability",
    ]
    audit_path.write_text("\n".join(audit), encoding="utf-8")

    print("\n=== AUDIT BACKTEST V2 ===")
    print("\n".join(audit))

    print("\n=== RETURN DAN STRATEGY MENURUT BIAS ===")
    print(summary.round(4).to_string(index=False))

    print("\n=== AKURASI ARAH ===")
    print(accuracy.round(4).to_string(index=False))

    print("\nFile hasil:")
    print(detail_path.relative_to(ROOT))
    print(summary_path.relative_to(ROOT))
    print(accuracy_path.relative_to(ROOT))
    print(audit_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
