
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

gold = pd.read_csv(RAW / "xauusd_d1.csv", parse_dates=["time"])
pairs = pd.read_csv(PROCESSED / "backtest_timing_pairs.csv")

gold = gold.sort_values("time").copy()
gold["weekday"] = gold["time"].dt.day_name()
gold["year"] = gold["time"].dt.year
gold["gap_days"] = gold["time"].diff().dt.days
gold["close_change_pct"] = gold["close"].pct_change() * 100

# 1. Frekuensi candle akhir pekan per tahun
print("\n=== 1. CANDLE MINGGU PER TAHUN ===")
sunday = gold[gold["weekday"] == "Sunday"]

print(
    sunday.groupby("year")
    .size()
    .rename("sunday_bars")
    .to_string()
)

# 2. Contoh candle Minggu dan candle di sekitarnya
print("\n=== 2. CONTOH CANDLE MINGGU ===")
indices = sunday.index.tolist()

if indices:
    sample_indices = indices[:5] + indices[len(indices)//2:len(indices)//2+5] + indices[-5:]
    sample_indices = sorted(set(sample_indices))

    for idx in sample_indices:
        pos = gold.index.get_loc(idx)
        start = max(0, pos - 1)
        end = min(len(gold), pos + 2)
        print(
            gold.iloc[start:end][
                ["time", "weekday", "open", "high", "low", "close",
                 "gap_days", "close_change_pct"]
            ].to_string(index=False)
        )
        print("-" * 55)

# 3. Distribusi hari dalam seluruh data
print("\n=== 3. DISTRIBUSI HARI ===")
print(gold["weekday"].value_counts().to_string())

# 4. Periksa kolom tanggal dan umur sinyal makro
print("\n=== 4. KOLOM PASANGAN WAKTU BACKTEST ===")
print("Kolom:", pairs.columns.tolist())

date_candidates = [
    c for c in pairs.columns
    if "date" in c.lower() or "time" in c.lower()
]
print(pairs[date_candidates].head(10).to_string(index=False))

if "macro_age_days" in pairs.columns:
    print("\n=== 5. CONTOH SINYAL BERUMUR 4-5 HARI ===")
    cols = [
        c for c in [
            "gold_date", "macro_date", "macro_age_days",
            "macro_score_v2", "macro_bias_v2",
            "dollar_observation_date", "dollar_age_days"
        ]
        if c in pairs.columns
    ]
    print(
        pairs[pairs["macro_age_days"] > 3][cols]
        .head(20)
        .to_string(index=False)
    )

print("\nAudit selesai. Data sumber tidak diubah.")
