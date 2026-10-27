from pathlib import Path
import argparse
import sys
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from xau.snr import build_signals, simulate_vectorized
from xau.metrics import summarize_trades, annual_report


def align_macro(df, macro_path):
    if not macro_path.exists():
        raise FileNotFoundError(f"Macro PIT file missing: {macro_path}")
    macro = pd.read_csv(macro_path)
    required = {"date", "macro_bias", "available_factors"}
    missing = required - set(macro.columns)
    if missing:
        raise ValueError(f"Macro PIT file missing columns: {sorted(missing)}")
    macro["date"] = pd.to_datetime(macro["date"], utc=True, errors="raise")
    macro["available_at"] = macro["date"] + pd.Timedelta(days=1)
    macro["available_factors"] = pd.to_numeric(macro["available_factors"], errors="coerce")
    if "macro_score" in macro:
        macro["macro_score"] = pd.to_numeric(macro["macro_score"], errors="coerce")
    macro = macro.sort_values("available_at")
    out = pd.merge_asof(df.sort_values("signal_time"), macro, left_on="signal_time",
                        right_on="available_at", direction="backward", allow_exact_matches=True)
    return out.drop(columns=["available_at"])


def main():
    parser = argparse.ArgumentParser(description="Development-only test: does PIT macro filtering improve SnR expectancy?")
    parser.add_argument("--confirm-baseline-reviewed", action="store_true",
                        help="Explicitly confirm SnR development metrics were reviewed and the baseline is frozen.")
    args = parser.parse_args()
    if not args.confirm_baseline_reviewed:
        raise SystemExit("Blocked: first review reports/snr_development_summary.csv; rerun with --confirm-baseline-reviewed.")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    val, d, s, ex = cfg["validation"], cfg["data"], cfg["strategy"], cfg["execution"]
    reports = ROOT / d["reports_dir"]
    baseline_report = reports / "snr_development_summary.csv"
    if not baseline_report.exists():
        raise SystemExit("Blocked: run scripts/run_snr_backtest.py and review its development report first.")
    baseline = pd.read_csv(baseline_report)
    required_baseline = {"strategy", "expectancy_R", "ci_low_R", "ci_high_R", "trades"}
    if not required_baseline.issubset(baseline.columns) or not baseline["strategy"].astype(str).str.startswith("development slippage=").all():
        raise SystemExit("Blocked: this is not a current development-only report with expectancy CI. Rerun the maintained baseline runner first.")
    raw_dir = ROOT / d["raw_dir"]
    df = build_signals(raw_dir, d["symbol_file"], left=s["pivot_left"], right=s["pivot_right"],
        max_age_days=s["max_pivot_age_days"], tolerance_atr=s["zone_tolerance_atr"],
        min_pivots=s["min_zone_pivots"], max_width_atr=s["max_zone_width_atr"],
        min_body_range_ratio=s["min_body_range_ratio"],
        spread_points_multiplier=ex["spread_points_multiplier"])
    df = align_macro(df, ROOT / d["processed_dir"] / "macro_score_pit_v1.csv")
    bias = df["macro_bias"].astype(str).str.lower()
    macro_dir = pd.Series(0, index=df.index, dtype=int)
    macro_dir.loc[bias.str.contains("bullish", na=False)] = 1
    macro_dir.loc[bias.str.contains("bearish", na=False)] = -1
    factors = pd.to_numeric(df["available_factors"], errors="coerce").fillna(0)
    filtered = df.copy()
    filtered["signal"] = filtered["signal"].where((macro_dir == filtered["signal"]) & (factors >= 1), 0)
    dev_end = pd.Timestamp(val["development_end"], tz="UTC") + pd.Timedelta(days=1)
    dev_cutoff = dev_end - pd.Timedelta(minutes=5 * s["max_hold_bars"])
    dev_years = range(int(df["time"].min().year), dev_end.year)
    # Never run this comparison on the sealed holdout.
    base_dev = df.loc[df["time"] < dev_end].copy()
    filt_dev = filtered.loc[filtered["time"] < dev_end].copy()
    summaries, annuals = [], []
    for slip in ex["slippage_usd_scenarios"]:
        for label, subset in (("SnR baseline", base_dev), ("SnR + PIT macro filter", filt_dev)):
            trades = simulate_vectorized(subset, slippage_usd=float(slip),
                atr_stop_mult=s["atr_stop_mult"], target_r=s["target_r"], max_hold_bars=s["max_hold_bars"])
            if not trades.empty:
                trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
                trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
                trades = trades[(trades["entry_time"] < dev_cutoff) & (trades["exit_time"] < dev_end)]
            summaries.append(summarize_trades(trades, label=f"{label}; slip={slip:.2f}",
                repetitions=val["bootstrap_repetitions"], confidence=val["bootstrap_confidence"],
                seed=val["bootstrap_seed"]))
            annuals.append(annual_report(trades, years=dev_years).assign(strategy=label, slippage_usd=float(slip)))
    pd.DataFrame(summaries).to_csv(reports / "snr_macro_filter_development.csv", index=False)
    pd.concat(annuals, ignore_index=True).to_csv(reports / "snr_macro_filter_annual.csv", index=False)
    print("DEVELOPMENT ONLY — no holdout was simulated or reported.")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Saved: reports/snr_macro_filter_development.csv and reports/snr_macro_filter_annual.csv")


if __name__ == "__main__":
    main()
