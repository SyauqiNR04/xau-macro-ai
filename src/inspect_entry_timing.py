import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "bt", "src/backtest_intraday_v1.py"
)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

df = bt.build_dataset()

cols = [
    "time", "signal_time", "open", "high", "low", "close",
    "signal", "entry_signal", "entry_atr",
    "h4_h4_ema50", "h4_h4_ema200",
    "h1_h1_ema20", "h1_h1_ema50",
    "m15_close", "m15_m15_ema20", "m15_m15_atr14",
    "m5_prev_high", "m5_prev_low"
]
cols = [c for c in cols if c in df.columns]

active = df.index[df["entry_signal"].ne(0)].tolist()
print("Jumlah entry_signal:", len(active))

if active:
    # Tampilkan beberapa sinyal pertama, termasuk baris sebelumnya.
    chosen = []
    for i in active[:5]:
        chosen.extend(range(max(0, i - 2), min(len(df), i + 2)))
    chosen = sorted(set(chosen))

    print(df.loc[chosen, cols].to_string(index=False))
