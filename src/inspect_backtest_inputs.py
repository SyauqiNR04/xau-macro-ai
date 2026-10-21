
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

GOLD_FILE = ROOT / "data" / "raw" / "xauusd_d1.csv"
MACRO_FILE = ROOT / "data" / "processed" / "macro_score_v2.csv"


def inspect_csv(path, date_col, required_cols):
    print(f"\n{'=' * 65}")
    print(f"FILE: {path.name}")
    print(f"Exists: {path.exists()}")

    if not path.exists():
        print("ERROR: File tidak ditemukan.")
        return None

    df = pd.read_csv(path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nDtypes:")
    print(df.dtypes.to_string())

    missing_cols = sorted(set(required_cols) - set(df.columns))
    if missing_cols:
        print(f"\nMISSING COLUMNS: {missing_cols}")
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    print(f"\nInvalid dates: {df[date_col].isna().sum()}")
    print(f"Duplicate dates: {df[date_col].duplicated().sum()}")
    print(f"Date range: {df[date_col].min()} -> {df[date_col].max()}")

    print("\nMissing values:")
    print(df[required_cols].isna().sum().to_string())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nLast 3 rows:")
    print(df.tail(3).to_string(index=False))

    return df


gold = inspect_csv(
    GOLD_FILE,
    date_col="time",
    required_cols=["time", "open", "high", "low", "close", "volume"],
)

macro = inspect_csv(
    MACRO_FILE,
    date_col="observation_date",
    required_cols=[
        "observation_date",
        "macro_score_v2",
        "macro_bias_v2",
        "signal_coverage",
    ],
)

if gold is not None and macro is not None:
    if "close" in gold.columns and "macro_score_v2" in macro.columns:
        gold["close"] = pd.to_numeric(gold["close"], errors="coerce")
        macro["macro_score_v2"] = pd.to_numeric(
            macro["macro_score_v2"], errors="coerce"
        )

        gold_dates = gold["time"].dropna()
        macro_dates = macro["observation_date"].dropna()

        start = max(gold_dates.min(), macro_dates.min())
        end = min(gold_dates.max(), macro_dates.max())

        print(f"\n{'=' * 65}")
        print("COMPATIBILITY CHECK")
        print(f"Gold rows: {len(gold):,}")
        print(f"Macro rows: {len(macro):,}")
        print(f"Overlapping date range: {start} -> {end}")

        if start <= end:
            gold_overlap = gold.loc[
                gold["time"].between(start, end), "time"
            ].dt.normalize()

            macro_overlap = macro.loc[
                macro["observation_date"].between(start, end),
                "observation_date",
            ].dt.normalize()

            print(f"Gold observations in overlap: {len(gold_overlap):,}")
            print(f"Macro observations in overlap: {len(macro_overlap):,}")
            print(
                "Unique overlapping calendar dates:",
                len(set(gold_overlap) & set(macro_overlap)),
            )
        else:
            print("WARNING: Tidak ada rentang tanggal yang tumpang tindih.")

        print(
            "Invalid/non-numeric gold close:",
            gold["close"].isna().sum(),
        )
        print(
            "Missing/non-numeric macro score:",
            macro["macro_score_v2"].isna().sum(),
        )
