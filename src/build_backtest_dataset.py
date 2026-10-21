
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

GOLD_FILE = ROOT / "data" / "raw" / "xauusd_d1.csv"
MACRO_FILE = ROOT / "data" / "processed" / "macro_score_v2.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "backtest_dataset.csv"

HORIZONS = [1, 3, 5]


def load_gold():
    df = pd.read_csv(GOLD_FILE)

    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom gold tidak ditemukan: {missing}")

    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df["close"] = pd.to_numeric(df["close"], errors="raise")

    df = df.sort_values("time").reset_index(drop=True)

    if df["time"].duplicated().any():
        raise ValueError("Tanggal gold duplikat ditemukan.")

    if df["close"].isna().any() or (df["close"] <= 0).any():
        raise ValueError("Close gold kosong atau tidak valid.")

    return df


def load_macro():
    df = pd.read_csv(MACRO_FILE)

    required = {
        "observation_date",
        "macro_score_v2",
        "available_factors",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom macro tidak ditemukan: {missing}")

    df["observation_date"] = pd.to_datetime(
        df["observation_date"], errors="raise"
    )
    df["macro_score_v2"] = pd.to_numeric(
        df["macro_score_v2"], errors="coerce"
    )

    df = df.sort_values("observation_date").reset_index(drop=True)

    if df["observation_date"].duplicated().any():
        raise ValueError("Tanggal macro duplikat ditemukan.")

    # Gunakan skor valid yang didukung kedua faktor.
    df = df.loc[
        df["macro_score_v2"].notna()
        & df["available_factors"].eq(2)
    ].copy()

    return df[
        [
            "observation_date",
            "macro_score_v2",
            "available_factors",
        ]
    ]


def main():
    gold = load_gold()
    macro = load_macro()

    # Return ke depan dihitung berdasarkan jumlah candle,
    # bukan jumlah hari kalender.
    for horizon in HORIZONS:
        gold[f"forward_return_{horizon}d"] = (
            gold["close"].shift(-horizon) / gold["close"] - 1
        )

    # Konservatif: skor harus berasal dari tanggal yang
    # benar-benar lebih awal daripada tanggal candle emas.
    merged = pd.merge_asof(
        gold.sort_values("time"),
        macro.sort_values("observation_date"),
        left_on="time",
        right_on="observation_date",
        direction="backward",
        allow_exact_matches=False,
    )

    merged = merged.rename(
        columns={
            "time": "gold_date",
            "observation_date": "macro_source_date",
        }
    )

    merged["macro_age_days"] = (
        merged["gold_date"] - merged["macro_source_date"]
    ).dt.days

    # Audit usia skor agar stale signal dapat dikenali.
    merged["macro_is_stale_3d"] = merged["macro_age_days"] > 3

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    valid = merged["macro_score_v2"].notna()

    print("=" * 65)
    print("BACKTEST DATASET BUILD REPORT")
    print("=" * 65)
    print(f"Gold rows             : {len(gold):,}")
    print(f"Valid macro rows      : {len(macro):,}")
    print(f"Matched gold rows     : {valid.sum():,}")
    print(f"Unmatched gold rows   : {(~valid).sum():,}")
    print(f"Stale signals (>3d)   : "
          f"{merged.loc[valid, 'macro_is_stale_3d'].sum():,}")
    print(f"Date range            : "
          f"{merged['gold_date'].min()} -> "
          f"{merged['gold_date'].max()}")
    print(f"Saved                 : {OUTPUT_FILE}")

    print("\nForward return availability:")
    for horizon in HORIZONS:
        col = f"forward_return_{horizon}d"
        print(f"{col}: {merged[col].notna().sum():,} rows")

    print("\nSample:")
    print(
        merged[
            [
                "gold_date",
                "close",
                "macro_source_date",
                "macro_score_v2",
                "macro_age_days",
                *[f"forward_return_{h}d" for h in HORIZONS],
            ]
        ].tail(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
