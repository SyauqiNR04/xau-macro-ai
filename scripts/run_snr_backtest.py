from pathlib import Path
import argparse
import sys
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xau.snr import build_signals, simulate_vectorized
from xau.metrics import summarize_trades, annual_report


def run_simulation(frame, cfg, slip):
    s = cfg["strategy"]
    return simulate_vectorized(frame, slippage_usd=float(slip), atr_stop_mult=s["atr_stop_mult"],
                               target_r=s["target_r"], max_hold_bars=s["max_hold_bars"])


def main():
    parser = argparse.ArgumentParser(description="SnR backtest; holdout sealed by default.")
    parser.add_argument("--final-unseal", action="store_true",
                        help="Reveal holdout metrics only after strategy/settings are frozen.")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    val, d, s, ex = cfg["validation"], cfg["data"], cfg["strategy"], cfg["execution"]
    reports = ROOT / d["reports_dir"]
    reports.mkdir(parents=True, exist_ok=True)
    df = build_signals(ROOT / d["raw_dir"], d["symbol_file"], left=s["pivot_left"], right=s["pivot_right"],
        max_age_days=s["max_pivot_age_days"], tolerance_atr=s["zone_tolerance_atr"],
        min_pivots=s["min_zone_pivots"], max_width_atr=s["max_zone_width_atr"],
        min_body_range_ratio=s["min_body_range_ratio"],
        spread_points_multiplier=ex["spread_points_multiplier"])
    dev_end = pd.Timestamp(val["development_end"], tz="UTC") + pd.Timedelta(days=1)
    hold_start = pd.Timestamp(val["holdout_start"], tz="UTC")
    dev_df = df.loc[df["time"] < dev_end].copy()
    print(f"Bars loaded: {len(df):,}; range {df['time'].min()} to {df['time'].max()}")
    print("Spread validation: PASS. Missing/invalid/all-zero spread causes a hard failure.")
    print(f"Development window ends before {dev_end}; holdout starts {hold_start.date()} and is sealed by default.")
    summaries, annuals, final_rows = [], [], []
    for slip in ex["slippage_usd_scenarios"]:
        dev_trades = run_simulation(dev_df, cfg, slip)
        if dev_trades.empty:
            dev_trades = pd.DataFrame(columns=["entry_time", "exit_time", "R"])
        dev_trades["entry_time"] = pd.to_datetime(dev_trades["entry_time"], utc=True)
        dev_trades["exit_time"] = pd.to_datetime(dev_trades["exit_time"], utc=True)
        dev_trades = dev_trades[(dev_trades["entry_time"] < dev_end) & (dev_trades["exit_time"] < dev_end)]
        summaries.append(summarize_trades(dev_trades, label=f"development slippage={slip:.2f}",
            repetitions=val["bootstrap_repetitions"], confidence=val["bootstrap_confidence"],
            seed=val["bootstrap_seed"]))
        annuals.append(annual_report(dev_trades).assign(slippage_usd=float(slip)))
        # Holdout simulator is not run at all unless both switches are explicitly enabled.
        if args.final_unseal and val.get("holdout_enabled", False):
            # Start the holdout simulation flat while retaining pre-holdout indicator/zone history.
            hold_df = df.loc[df["time"] >= hold_start].copy()
            hold = run_simulation(hold_df, cfg, slip)
            if not hold.empty:
                hold["entry_time"] = pd.to_datetime(hold["entry_time"], utc=True)
            final_rows.append(summarize_trades(hold, label=f"FINAL HOLDOUT slippage={slip:.2f}",
                repetitions=val["bootstrap_repetitions"], confidence=val["bootstrap_confidence"],
                seed=val["bootstrap_seed"]))
    pd.DataFrame(summaries).to_csv(reports / "snr_development_summary.csv", index=False)
    pd.concat(annuals, ignore_index=True).to_csv(reports / "snr_development_annual.csv", index=False)
    print("\n=== DEVELOPMENT ONLY: SLIPPAGE SENSITIVITY + EXPECTANCY BOOTSTRAP CI ===")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Annual trade report saved: reports/snr_development_annual.csv")
    if args.final_unseal and val.get("holdout_enabled", False):
        pd.DataFrame(final_rows).to_csv(reports / "snr_holdout_final.csv", index=False)
        print("\n=== FINAL HOLDOUT ===")
        print(pd.DataFrame(final_rows).to_string(index=False))
        hold_annual = []
        for slip in ex["slippage_usd_scenarios"]:
            hold_df = df.loc[df["time"] >= hold_start].copy()
            hold_trades = run_simulation(hold_df, cfg, slip)
            hold_annual.append(annual_report(hold_trades).assign(slippage_usd=float(slip)))
        pd.concat(hold_annual, ignore_index=True).to_csv(reports / "snr_holdout_annual_final.csv", index=False)
    else:
        print("HOLDOUT SEALED: its simulator was not run and no holdout metrics were written.")


if __name__ == "__main__":
    main()
