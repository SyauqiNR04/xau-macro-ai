from pathlib import Path
import pandas as pd

raw = Path("data/raw")
specs = {
    "M5": ("xauusdm_m5.csv", 5),
    "M15": ("xauusdm_m15.csv", 15),
    "H1": ("xauusdm_h1.csv", 60),
    "H4": ("xauusdm_h4.csv", 240),
}

frames = {}
for name, (filename, minutes) in specs.items():
    df = pd.read_csv(raw / filename)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df["available_at"] = df["time"] + pd.Timedelta(minutes=minutes)
    frames[name] = df

print("=== RENTANG DAN JARAK WAKTU ===")
for name, (_, minutes) in specs.items():
    df = frames[name]
    delta = df["time"].diff().dt.total_seconds().div(60)
    print(
        f"{name}: rows={len(df)}, "
        f"first={df['time'].iloc[0]}, "
        f"last={df['time'].iloc[-1]}, "
        f"median_delta_min={delta.median()}, "
        f"available_at_offset_min={minutes}"
    )

print("\n=== CONTOH WAKTU CANDLE ===")
for name in ["M5", "M15", "H1", "H4"]:
    df = frames[name]
    print(f"\n{name}")
    print(df[["time", "available_at"]].head(5).to_string(index=False))

print("\n=== CONTOH SINKRONISASI M5 KE TIMEFRAME BESAR ===")
m5 = frames["M5"][["time"]].copy()
m5["signal_time"] = m5["time"] + pd.Timedelta(minutes=5)

for name in ["M15", "H1", "H4"]:
    higher = frames[name][["time", "available_at"]].copy()
    higher = higher.rename(columns={
        "time": f"{name.lower()}_open",
        "available_at": f"{name.lower()}_available"
    })

    merged = pd.merge_asof(
        m5.sort_values("signal_time"),
        higher.sort_values(f"{name.lower()}_available"),
        left_on="signal_time",
        right_on=f"{name.lower()}_available",
        direction="backward",
        allow_exact_matches=True
    )

    valid = merged[f"{name.lower()}_available"].notna()
    future = (
        merged.loc[valid, f"{name.lower()}_available"]
        > merged.loc[valid, "signal_time"]
    )

    print(f"\n{name}:")
    print("Rows without higher-TF candle:", int((~valid).sum()))
    print("Higher-TF availability after signal time:", int(future.sum()))
    print(merged.tail(5).to_string(index=False))
