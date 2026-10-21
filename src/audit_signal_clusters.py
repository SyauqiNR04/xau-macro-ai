import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "bt", "src/backtest_intraday_v1.py"
)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

df = bt.build_dataset()

for col in ["signal", "entry_signal"]:
    s = df[col].fillna(0).astype(int)
    active = s.ne(0)
    new_cluster = active & (
        s.shift(1).fillna(0).astype(int).ne(s)
    )

    print(f"\n=== {col} ===")
    print("Total sinyal:", int(active.sum()))
    print("Awal kelompok sinyal:", int(new_cluster.sum()))

    for direction, label in [(1, "LONG"), (-1, "SHORT")]:
        mask = s.eq(direction)
        starts = mask & ~mask.shift(1, fill_value=False)
        print(
            f"{label}: total={int(mask.sum())}, "
            f"kelompok={int(starts.sum())}"
        )

    # Jarak antar-sinyal aktif, dihitung dalam jumlah candle M5
    idx = df.index[active]
    if len(idx) > 1:
        gaps = pd.Series(idx).diff().dropna()
        print("Median jarak sinyal (candle M5):", gaps.median())
        print("Sinyal berjarak <= 3 candle:", int((gaps <= 3).sum()))
