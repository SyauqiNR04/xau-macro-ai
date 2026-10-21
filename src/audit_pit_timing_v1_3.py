from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

BACKTEST = PROCESSED / "backtest_pit_v1.csv"
XAU = RAW / "xauusd_d1.csv"
FEATURES = PROCESSED / "fred_pit_daily_features_conservative.csv"
SCORE = PROCESSED / "macro_score_pit_v1.csv"

def parse_dt(series):
    # Handle ordinary date strings and Unix epoch seconds/milliseconds.
    numeric = pd.to_numeric(series, errors="coerce")
    if len(series) and numeric.notna().mean() > 0.8:
        med = numeric.dropna().abs().median()
        unit = "ms" if med > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)

def read_optional(path, label):
    if not path.exists():
        print(f"[MISSING] {label}: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[OK] {label}: {len(df):,} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"[ERROR] {label}: {e}")
        return None

def date_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def pct(x):
    return f"{x:.6f}%" if pd.notna(x) else "NA"

def main():
    print("=== PIT v1.3 TIMING & DATA ALIGNMENT AUDIT ===")
    print(f"Project root: {ROOT}")
    bt = read_optional(BACKTEST, "Backtest dataset")
    xau = read_optional(XAU, "Raw XAUUSD D1")
    feat = read_optional(FEATURES, "Conservative FRED PIT features")
    score = read_optional(SCORE, "PIT macro score")

    if bt is None:
        raise FileNotFoundError(f"Backtest dataset tidak ditemukan: {BACKTEST}")

    # Parse primary backtest timeline.
    bt_time_col = date_col(bt, ["time", "date", "datetime", "timestamp"])
    if bt_time_col is None:
        raise ValueError(f"Kolom waktu tidak ditemukan di backtest. Kolom: {list(bt.columns)}")
    bt["audit_timestamp_utc"] = parse_dt(bt[bt_time_col])
    bt["audit_date"] = bt["audit_timestamp_utc"].dt.strftime("%Y-%m-%d")
    bt = bt.sort_values("audit_timestamp_utc").reset_index(drop=True)

    print("\n--- 1. BACKTEST TIMELINE ---")
    print(f"Time column: {bt_time_col}")
    print(f"Rows: {len(bt):,}")
    print(f"Invalid timestamps: {bt['audit_timestamp_utc'].isna().sum():,}")
    print(f"Date range: {bt['audit_date'].min()} -> {bt['audit_date'].max()}")
    print(f"Duplicate timestamps: {bt['audit_timestamp_utc'].duplicated().sum():,}")
    print("Most common weekdays (UTC timestamp):")
    print(bt["audit_timestamp_utc"].dt.day_name().value_counts().to_string())
    print("Timestamp hour distribution (UTC; only meaningful if source has intraday time):")
    print(bt["audit_timestamp_utc"].dt.hour.value_counts().sort_index().to_string())

    # Find actual raw XAU timestamp/date convention and compare.
    if xau is not None:
        x_time_col = date_col(xau, ["time", "date", "datetime", "timestamp"])
        print("\n--- 2. RAW XAUUSD SOURCE ---")
        print(f"Columns: {list(xau.columns)}")
        if x_time_col:
            xau["audit_timestamp_utc"] = parse_dt(xau[x_time_col])
            xau["audit_date"] = xau["audit_timestamp_utc"].dt.strftime("%Y-%m-%d")
            print(f"Time column: {x_time_col}")
            print(f"Date range: {xau['audit_date'].min()} -> {xau['audit_date'].max()}")
            print(f"Invalid timestamps: {xau['audit_timestamp_utc'].isna().sum():,}")
            print(f"Duplicate timestamps: {xau['audit_timestamp_utc'].duplicated().sum():,}")
            print("Raw XAU columns used for return validation:")
            print([c for c in ["open", "high", "low", "close", "volume"] if c in xau.columns])
            if "close" in xau.columns:
                xau["close"] = pd.to_numeric(xau["close"], errors="coerce")
                xau = xau.sort_values("audit_timestamp_utc").reset_index(drop=True)
                for h in (1, 3, 5):
                    calc = (xau["close"].shift(-h) / xau["close"] - 1) * 100
                    print(f"Recomputed close-to-close return h={h}: {calc.notna().sum():,} valid values")
                    bt_ret_col = f"forward_return_{h}bar_pct"
                    if bt_ret_col in bt.columns:
                        chk = bt[["audit_date", bt_ret_col]].copy()
                        chk[bt_ret_col] = pd.to_numeric(chk[bt_ret_col], errors="coerce")
                        raw_map = pd.DataFrame({
                            "audit_date": xau["audit_date"],
                            f"raw_recomputed_h{h}": calc
                        })
                        chk = chk.merge(raw_map, on="audit_date", how="inner")
                        delta = (chk[bt_ret_col] - chk[f"raw_recomputed_h{h}"]).abs()
                        print(
                            f"  Compare with backtest h={h}: n={delta.notna().sum():,}, "
                            f"median abs diff={delta.median():.8f} pp, "
                            f"max abs diff={delta.max():.8f} pp"
                        )
        else:
            print("Tidak menemukan kolom tanggal/waktu di raw XAU.")

    # Compare DFII observation date against backtest date and reported age.
    print("\n--- 3. REAL-YIELD AGE CONSISTENCY ---")
    obs_col = date_col(bt, [
        "latest_observation_date_DFII10", "observation_date_DFII10",
        "DFII10_observation_date", "latest_observation_date"
    ])
    age_col = date_col(bt, ["real_yield_age_days", "yield_age_days"])
    if obs_col:
        bt["audit_obs_timestamp"] = parse_dt(bt[obs_col])
        bt["audit_obs_date"] = bt["audit_obs_timestamp"].dt.strftime("%Y-%m-%d")
        bt["audit_calendar_age"] = (
            bt["audit_timestamp_utc"].dt.normalize() -
            bt["audit_obs_timestamp"].dt.normalize()
        ).dt.days
        print(f"Observation date column: {obs_col}")
        print(f"Missing observation dates: {bt['audit_obs_timestamp'].isna().sum():,}")
        print("Calendar-age distribution:")
        print(bt["audit_calendar_age"].value_counts(dropna=False).sort_index().to_string())
        if age_col:
            bt[age_col] = pd.to_numeric(bt[age_col], errors="coerce")
            comparable = bt[["audit_calendar_age", age_col]].dropna()
            mismatch = comparable[comparable["audit_calendar_age"] != comparable[age_col]]
            print(f"Reported age column: {age_col}")
            print(f"Comparable rows: {len(comparable):,}")
            print(f"Rows where reported age differs from calendar-date difference: {len(mismatch):,}")
            if len(mismatch):
                print(mismatch.head(10).to_string(index=False))
    else:
        print("Kolom tanggal observasi DFII10 tidak ditemukan di backtest.")
        print(f"Columns: {list(bt.columns)}")

    # Compare score/features files and report date overlap/score alignment.
    print("\n--- 4. SCORE / FEATURE DATE ALIGNMENT ---")
    for label, data in [("PIT score", score), ("PIT features", feat)]:
        if data is None:
            continue
        dcol = date_col(data, ["date", "time", "observation_date", "timestamp"])
        if dcol is None:
            print(f"{label}: kolom tanggal tidak ditemukan. Columns={list(data.columns)}")
            continue
        data["audit_date"] = parse_dt(data[dcol]).dt.strftime("%Y-%m-%d")
        dates = set(data["audit_date"].dropna())
        bt_dates = set(bt["audit_date"].dropna())
        overlap = len(dates & bt_dates)
        print(
            f"{label}: date_col={dcol}, rows={len(data):,}, "
            f"unique dates={len(dates):,}, overlap with backtest dates={overlap:,}, "
            f"only source dates={len(dates - bt_dates):,}, only backtest dates={len(bt_dates - dates):,}"
        )
        print(f"{label} columns: {list(data.columns)}")

    # Shift sensitivity: same score on candle date vs adjacent available candle rows.
    print("\n--- 5. DATE-SHIFT SENSITIVITY (DIAGNOSTIC ONLY) ---")
    score_col = date_col(bt, ["macro_score"])
    if score_col:
        bt[score_col] = pd.to_numeric(bt[score_col], errors="coerce")
        age = pd.to_numeric(bt[age_col], errors="coerce") if age_col else pd.Series(np.nan, index=bt.index)
        eligible_base = bt[score_col].notna() & (bt[score_col] != 0)
        if age_col:
            eligible_base &= age.between(0, 3)
        for h in (1, 3, 5):
            rcol = f"forward_return_{h}bar_pct"
            if rcol not in bt.columns:
                continue
            ret = pd.to_numeric(bt[rcol], errors="coerce")
            print(f"Horizon {h}:")
            for shift in (-2, -1, 0, 1, 2):
                # shift > 0 uses a score from earlier rows for each return row.
                shifted_score = bt[score_col].shift(shift)
                ok = eligible_base & shifted_score.notna() & (shifted_score != 0) & ret.notna()
                strat = np.sign(shifted_score[ok]) * ret[ok]
                edge_vs_long = strat - ret[ok]
                print(
                    f"  score_shift={shift:+d} bars; n={int(ok.sum())}; "
                    f"strategy_mean={pct(strat.mean())}; "
                    f"edge_vs_long={pct(edge_vs_long.mean())}"
                )
        print("Interpretation: shift sensitivity does NOT establish the correct entry alignment.")
    else:
        print("Kolom macro_score tidak ditemukan.")

    # Export a row-level audit table without altering original source files.
    keep = [c for c in [
        "audit_timestamp_utc", "audit_date", "macro_score", "macro_bias",
        "change_5obs_DFII10", obs_col, age_col, "audit_calendar_age",
        "cpi_yoy_context", "signal_eligible",
        "forward_return_1bar_pct", "forward_return_3bar_pct", "forward_return_5bar_pct"
    ] if c and c in bt.columns]
    audit_path = PROCESSED / "pit_v1_3_timing_row_audit.csv"
    bt[keep].to_csv(audit_path, index=False)

    summary = {
        "rows": len(bt),
        "date_min": bt["audit_date"].min(),
        "date_max": bt["audit_date"].max(),
        "invalid_timestamps": int(bt["audit_timestamp_utc"].isna().sum()),
        "duplicate_timestamps": int(bt["audit_timestamp_utc"].duplicated().sum()),
        "time_column": bt_time_col,
        "observation_date_column": obs_col or "",
        "age_column": age_col or "",
        "warning": "Audit diagnostics only; exact release availability and broker candle session must be verified externally."
    }
    summary_path = PROCESSED / "pit_v1_3_timing_audit_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n=== OUTPUTS ===")
    print(audit_path)
    print(summary_path)
    print("\nAudit selesai. Source files tidak diubah.")
    print("PENTING: umur kalender 1 hari tidak otomatis membuktikan data sudah dipublikasikan sebelum candle entry.")

if __name__ == "__main__":
    main()
