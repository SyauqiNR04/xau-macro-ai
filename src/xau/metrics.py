from __future__ import annotations
import numpy as np
import pandas as pd

def expectancy_bootstrap_ci(values, *, repetitions=5000, confidence=0.95, seed=20261027):
    """Percentile bootstrap CI for mean trade R."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": int(len(x)), "expectancy_R": float(x.mean()) if len(x) else np.nan,
                "ci_low_R": np.nan, "ci_high_R": np.nan}
    rng = np.random.default_rng(seed)
    means = np.empty(int(repetitions), dtype=float)
    chunk = max(1, min(256, int(repetitions)))
    done = 0
    while done < repetitions:
        n = min(chunk, repetitions - done)
        means[done:done+n] = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
        done += n
    alpha = 1.0 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"n": int(len(x)), "expectancy_R": float(x.mean()), "ci_low_R": float(lo), "ci_high_R": float(hi)}

def summarize_trades(trades, *, label, repetitions=5000, confidence=0.95, seed=20261027):
    if trades.empty:
        return {"strategy": label, "trades": 0, "win_rate_pct": np.nan, "expectancy_R": np.nan,
                "ci_low_R": np.nan, "ci_high_R": np.nan, "profit_factor": np.nan,
                "net_R": 0.0, "max_drawdown_R": np.nan}
    r = pd.to_numeric(trades["R"], errors="coerce").dropna().to_numpy()
    ci = expectancy_bootstrap_ci(r, repetitions=repetitions, confidence=confidence, seed=seed)
    equity = np.cumsum(r)
    dd = equity - np.maximum.accumulate(equity)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return {"strategy": label, "trades": len(r), "win_rate_pct": float((r > 0).mean()*100),
            "expectancy_R": ci["expectancy_R"], "ci_low_R": ci["ci_low_R"], "ci_high_R": ci["ci_high_R"],
            "profit_factor": float(gp/gl) if gl else np.inf, "net_R": float(r.sum()),
            "max_drawdown_R": float(dd.min())}

def annual_report(trades):
    cols = ["year", "trades", "win_rate_pct", "expectancy_R", "net_R", "profit_factor"]
    if trades.empty:
        return pd.DataFrame(columns=cols)
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="raise")
    df["year"] = df.entry_time.dt.year
    rows = []
    for year, group in df.groupby("year", sort=True):
        r = pd.to_numeric(group["R"], errors="coerce").dropna()
        gp, gl = r[r > 0].sum(), -r[r < 0].sum()
        rows.append({"year": int(year), "trades": int(len(r)),
                     "win_rate_pct": float((r > 0).mean()*100) if len(r) else np.nan,
                     "expectancy_R": float(r.mean()) if len(r) else np.nan,
                     "net_R": float(r.sum()), "profit_factor": float(gp/gl) if gl else np.inf})
    return pd.DataFrame(rows)
