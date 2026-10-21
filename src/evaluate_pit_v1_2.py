from pathlib import Path
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "backtest_pit_v1.csv"
OUTDIR = ROOT / "data" / "processed"

HORIZONS = (1, 3, 5)
N_BOOT = 3000
BLOCK_LENGTH = 10
SEED = 20261021

def pct(x):
    return float(x) if pd.notna(x) else np.nan

def parse_dates(series):
    # Supports ISO dates and Unix timestamps (seconds or milliseconds).
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        med = numeric.dropna().abs().median()
        unit = "ms" if med > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True).dt.tz_convert(None)
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)

def block_bootstrap_ci(values, block_len=10, n_boot=3000, seed=20261021):
    """
    Circular moving-block bootstrap over the original candle timeline.
    NaNs (days without an eligible signal) remain in the sampled timeline.
    Returns percentile CI for the mean of finite values.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0 or np.isfinite(x).sum() < 20:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    L = max(1, min(int(block_len), n))
    n_blocks = int(np.ceil(n / L))
    starts = np.arange(n)
    estimates = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        chosen_starts = rng.choice(starts, size=n_blocks, replace=True)
        idx = np.concatenate([(s + np.arange(L)) % n for s in chosen_starts])[:n]
        sample = x[idx]
        sample = sample[np.isfinite(sample)]
        estimates[b] = sample.mean() if len(sample) else np.nan

    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) < 100:
        return np.nan, np.nan
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)

def metric_row(frame, horizon, label, benchmark_name, strategy_col, benchmark_col):
    valid = frame[strategy_col].notna() & frame[benchmark_col].notna()
    x = frame.loc[valid, strategy_col].astype(float)
    b = frame.loc[valid, benchmark_col].astype(float)
    edge = x - b
    # Preserve gaps in the full timeline for block bootstrap.
    timeline = pd.Series(np.nan, index=frame.index, dtype=float)
    timeline.loc[valid] = edge.to_numpy()
    lo, hi = block_bootstrap_ci(
        timeline.to_numpy(), BLOCK_LENGTH, N_BOOT,
        SEED + horizon * 100 + sum(ord(c) for c in label + benchmark_name)
    )
    direction_col = f"direction_correct_{horizon}bar"
    acc = np.nan
    if direction_col in frame.columns:
        d = pd.to_numeric(frame.loc[valid, direction_col], errors="coerce")
        if d.notna().any():
            acc = d.mean() * 100

    return {
        "split": label,
        "horizon_bars": horizon,
        "benchmark": benchmark_name,
        "n_signals": int(valid.sum()),
        "directional_accuracy_pct": acc,
        "strategy_mean_return_pct": x.mean() if len(x) else np.nan,
        "strategy_median_return_pct": x.median() if len(x) else np.nan,
        "strategy_win_rate_pct": (x > 0).mean() * 100 if len(x) else np.nan,
        "benchmark_mean_return_pct": b.mean() if len(b) else np.nan,
        "mean_edge_pct": edge.mean() if len(edge) else np.nan,
        "edge_ci95_low_pct": lo,
        "edge_ci95_high_pct": hi,
    }

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT}\n"
            "Pastikan backtest_pit_v1.csv ada di data/processed."
        )

    df = pd.read_csv(INPUT)
    required = ["time", "macro_score", "real_yield_age_days"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df["date_parsed"] = parse_dates(df["time"])
    if df["date_parsed"].isna().all():
        raise ValueError("Kolom 'time' gagal dibaca sebagai tanggal.")
    df = df.sort_values("date_parsed").reset_index(drop=True)
    df["original_bar_index"] = np.arange(len(df))

    for c in ["macro_score", "real_yield_age_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "signal_eligible" in df.columns:
        elig = df["signal_eligible"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        elig = pd.Series(True, index=df.index)

    eligible = (
        elig
        & df["macro_score"].notna()
        & (df["macro_score"] != 0)
        & df["real_yield_age_days"].between(0, 3, inclusive="both")
    )
    df["eligible_signal"] = eligible
    df["signal_direction"] = np.sign(df["macro_score"])

    summary_rows = []
    annual_rows = []
    walk_rows = []
    nonoverlap_rows = []

    # Fixed, predeclared splits; no parameter tuning on the final test period.
    splits = [
        ("development_2015_2021", "2015-01-01", "2021-12-31"),
        ("validation_2022_2024", "2022-01-01", "2024-12-31"),
        ("test_2025_2026", "2025-01-01", "2026-12-31"),
    ]

    for h in HORIZONS:
        ret_col = f"forward_return_{h}bar_pct"
        if ret_col not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {ret_col}")
        df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")

        # Strategy: long for positive macro score, short for negative score.
        df[f"strategy_{h}"] = np.where(
            df["eligible_signal"], df["signal_direction"] * df[ret_col], np.nan
        )
        df[f"long_{h}"] = np.where(df["eligible_signal"], df[ret_col], np.nan)
        df[f"short_{h}"] = np.where(df["eligible_signal"], -df[ret_col], np.nan)
        df[f"cash_{h}"] = np.where(df["eligible_signal"], 0.0, np.nan)

        for bench, bcol in [
            ("always_long", f"long_{h}"),
            ("always_short", f"short_{h}"),
            ("cash_no_return", f"cash_{h}"),
        ]:
            summary_rows.append(metric_row(
                df, h, "all_period", bench, f"strategy_{h}", bcol
            ))

        # Yearly metrics against cash and the long-only baseline.
        df["year"] = df["date_parsed"].dt.year
        for year, part in df.groupby("year", sort=True):
            for bench, bcol in [
                ("always_long", f"long_{h}"),
                ("cash_no_return", f"cash_{h}"),
            ]:
                annual_rows.append(metric_row(
                    part, h, str(year), bench, f"strategy_{h}", bcol
                ))

        for label, start, end in splits:
            part = df[df["date_parsed"].between(start, end, inclusive="both")].copy()
            for bench, bcol in [
                ("always_long", f"long_{h}"),
                ("always_short", f"short_{h}"),
                ("cash_no_return", f"cash_{h}"),
            ]:
                walk_rows.append(metric_row(
                    part, h, label, bench, f"strategy_{h}", bcol
                ))

        # Non-overlap subsets using the original candle index, not filtered row order.
        for offset in range(h):
            part = df[(df["original_bar_index"] % h == offset)].copy()
            for bench, bcol in [
                ("always_long", f"long_{h}"),
                ("cash_no_return", f"cash_{h}"),
            ]:
                row = metric_row(part, h, f"nonoverlap_offset_{offset}", bench,
                                 f"strategy_{h}", bcol)
                row["offset"] = offset
                nonoverlap_rows.append(row)

    # Timing diagnostics: these are checks, not proof of exact intraday availability.
    timing = {
        "rows_total": int(len(df)),
        "date_min": str(df["date_parsed"].min().date()),
        "date_max": str(df["date_parsed"].max().date()),
        "eligible_signal_rows": int(df["eligible_signal"].sum()),
        "rows_missing_date": int(df["date_parsed"].isna().sum()),
        "rows_missing_macro_score": int(df["macro_score"].isna().sum()),
        "rows_missing_yield_age": int(df["real_yield_age_days"].isna().sum()),
        "eligible_age_0": int((df["eligible_signal"] & (df["real_yield_age_days"] == 0)).sum()),
        "eligible_age_1_to_3": int((df["eligible_signal"] & df["real_yield_age_days"].between(1, 3)).sum()),
        "timing_caveat": (
            "Age eligibility is not proof that the signal was known before entry. "
            "Confirm XAUUSD candle timezone/session and official publication timestamps."
        ),
        "bootstrap_method": "circular moving-block bootstrap on full candle timeline",
        "bootstrap_block_length_bars": BLOCK_LENGTH,
        "bootstrap_replicates": N_BOOT,
        "costs_included": False,
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "pit_v1_2_block_bootstrap_summary.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(OUTDIR / "pit_v1_2_annual.csv", index=False)
    pd.DataFrame(walk_rows).to_csv(OUTDIR / "pit_v1_2_walk_forward.csv", index=False)
    pd.DataFrame(nonoverlap_rows).to_csv(OUTDIR / "pit_v1_2_nonoverlap.csv", index=False)
    pd.DataFrame([timing]).to_csv(OUTDIR / "pit_v1_2_timing_audit.csv", index=False)

    print("\n=== PIT v1.2: BLOCK BOOTSTRAP VS BENCHMARKS ===")
    print(pd.DataFrame(summary_rows).round(4).to_string(index=False))
    print("\n=== WALK-FORWARD SPLITS ===")
    print(pd.DataFrame(walk_rows).round(4).to_string(index=False))
    print("\n=== TIMING / DATA AUDIT ===")
    for k, v in timing.items():
        print(f"{k}: {v}")
    print("\nFile tersimpan:")
    for name in [
        "pit_v1_2_block_bootstrap_summary.csv",
        "pit_v1_2_annual.csv",
        "pit_v1_2_walk_forward.csv",
        "pit_v1_2_nonoverlap.csv",
        "pit_v1_2_timing_audit.csv",
    ]:
        print(OUTDIR / name)
    print("\nPeringatan:")
    print("- Hasil ini tetap simulasi close-to-close dan belum memasukkan biaya transaksi.")
    print("- Bootstrap blok membantu dependensi serial, tetapi panjang blok 10 bar adalah asumsi.")
    print("- Walk-forward di sini mengevaluasi aturan tetap; bukan bukti strategi sudah optimal.")
    print("- Pastikan timestamp rilis makro mendahului waktu entry yang realistis.")

if __name__ == "__main__":
    main()
