import importlib.util
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "bt", "src/backtest_intraday_v1.py"
)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

OUT = Path("reports")
OUT.mkdir(exist_ok=True)
OOS_START = pd.Timestamp("2024-01-01", tz="UTC")


def apply_cooldown(df, bars):
    out = df.copy()
    signals = out["entry_signal"].fillna(0).astype(int).to_numpy()
    filtered = signals.copy()
    last_accepted = {1: -10**12, -1: -10**12}

    for i, direction in enumerate(signals):
        if direction == 0:
            continue

        if i - last_accepted[direction] <= bars:
            filtered[i] = 0
        else:
            last_accepted[direction] = i

    out["entry_signal"] = filtered
    return out


df = bt.build_dataset()
print("Jumlah candle:", len(df))
print("Rentang:", df["time"].min(), "sampai", df["time"].max())

scenarios = [
    ("Baseline", None),
    ("Cooldown_3", 3),
    ("Cooldown_6", 6),
]

summary_rows = []
trade_frames = []

for scenario, bars in scenarios:
    test_df = df if bars is None else apply_cooldown(df, bars)

    for use_macro in [False, True]:
        label_base = "Macro" if use_macro else "Technical"

        trades = bt.simulate(test_df, use_macro)
        if trades is None:
            print("Simulator mengembalikan None:", scenario, label_base)
            continue

        trades = trades.copy()
        label = f"{label_base} | {scenario}"

        if trades.empty:
            print("Tidak ada transaksi:", label)
            continue

        # Simpan seluruh transaksi agar dapat diaudit ulang.
        trades["scenario"] = scenario
        trades["model"] = label_base
        trade_frames.append(trades)

        full = bt.summarize(trades, f"{label} | FULL")
        summary_rows.append(full)

        # OOS dipisahkan berdasarkan waktu entry transaksi.
        time_col = next(
            (c for c in ["entry_time", "entry_at", "entry_datetime"]
             if c in trades.columns),
            None
        )

        if time_col is None:
            print("\nKolom transaksi:", trades.columns.tolist())
            raise RuntimeError(
                "Kolom waktu entry tidak ditemukan. "
                "Perlu menyesuaikan nama kolom OOS."
            )

        entry_times = pd.to_datetime(trades[time_col], utc=True)
        oos = trades.loc[entry_times >= OOS_START].copy()
        summary_rows.append(
            bt.summarize(oos, f"{label} | OOS 2024+")
        )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "cooldown_comparison_summary.csv", index=False)

if trade_frames:
    all_trades = pd.concat(trade_frames, ignore_index=True)
    all_trades.to_csv(
        OUT / "cooldown_comparison_trades.csv", index=False
    )

print("\n=== PERBANDINGAN COOLDOWN ===")
print(summary.to_string(index=False))
print("\nLaporan disimpan:")
print(OUT / "cooldown_comparison_summary.csv")
print(OUT / "cooldown_comparison_trades.csv")
