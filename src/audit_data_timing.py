
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

GOLD_FILE = RAW / "xauusd_d1.csv"
MACRO_FILE = PROCESSED / "macro_score_v2.csv"
BACKTEST_FILE = PROCESSED / "backtest_dataset.csv"

OUTPUT_DIR = PROCESSED
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path, date_columns):
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path)

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def main():
    gold = load_csv(GOLD_FILE, ["time"])
    macro = load_csv(
        MACRO_FILE,
        ["observation_date", "dollar_observation_date"]
    )
    backtest = load_csv(
        BACKTEST_FILE,
        ["gold_date", "macro_date"]
    )

    # --------------------------------------------------
    # 1. AUDIT CANDLE XAUUSD
    # --------------------------------------------------
    gold["time"] = pd.to_datetime(gold["time"], errors="coerce")
    gold = gold.dropna(subset=["time"]).sort_values("time").copy()

    gold["date_gap_days"] = gold["time"].diff().dt.days
    gold["weekday"] = gold["time"].dt.day_name()

    section("1. AUDIT DATA XAUUSD")
    print(f"Jumlah candle       : {len(gold)}")
    print(f"Tanggal awal        : {gold['time'].min()}")
    print(f"Tanggal akhir       : {gold['time'].max()}")
    print(f"Tanggal duplikat    : {gold['time'].duplicated().sum()}")
    print(f"Tanggal tidak unik  : {gold['time'].dt.normalize().duplicated().sum()}")
    print(f"Bar hari Minggu     : {(gold['weekday'] == 'Sunday').sum()}")
    print(f"Bar hari Sabtu      : {(gold['weekday'] == 'Saturday').sum()}")
    print("\nDistribusi jeda antarbar (hari kalender):")
    print(gold["date_gap_days"].value_counts(dropna=False).sort_index().to_string())

    gold[gold["weekday"].isin(["Saturday", "Sunday"])].to_csv(
        OUTPUT_DIR / "gold_weekend_bars.csv", index=False
    )

    # --------------------------------------------------
    # 2. AUDIT SKOR MAKRO
    # --------------------------------------------------
    macro = macro.sort_values("observation_date").copy()

    section("2. AUDIT SKOR MAKRO")
    print(f"Jumlah baris        : {len(macro)}")
    print(f"Tanggal awal        : {macro['observation_date'].min()}")
    print(f"Tanggal akhir       : {macro['observation_date'].max()}")
    print(f"Tanggal duplikat    : {macro['observation_date'].duplicated().sum()}")

    if "available_factors" in macro.columns:
        print("\nJumlah faktor tersedia:")
        print(macro["available_factors"].value_counts(dropna=False).sort_index().to_string())

    if "macro_score_v2" in macro.columns:
        print(f"\nSkor makro kosong   : {macro['macro_score_v2'].isna().sum()}")

        valid = macro.dropna(subset=["macro_score_v2"]).copy()
        valid["score_change"] = valid["macro_score_v2"].ne(
            valid["macro_score_v2"].shift()
        )

        print(
            "Baris skor sama dengan baris valid sebelumnya: "
            f"{(~valid['score_change']).sum()}"
        )

    if "dollar_age_days" in macro.columns:
        print("\nUmur data dolar (hari):")
        print(macro["dollar_age_days"].describe().round(2).to_string())

    # --------------------------------------------------
    # 3. AUDIT PEMASANGAN SKOR KE CANDLE EMAS
    # --------------------------------------------------
    section("3. AUDIT DATASET BACKTEST")

    required = {"gold_date", "macro_age_days", "macro_score_v2"}
    missing = required - set(backtest.columns)

    if missing:
        print(f"Kolom yang dibutuhkan tidak ditemukan: {sorted(missing)}")
        print(f"Kolom tersedia: {backtest.columns.tolist()}")
        return

    backtest = backtest.sort_values("gold_date").copy()

    print(f"Jumlah baris        : {len(backtest)}")
    print(f"Tanggal emas kosong : {backtest['gold_date'].isna().sum()}")
    print(f"Skor makro kosong   : {backtest['macro_score_v2'].isna().sum()}")

    print("\nUmur skor saat dipasangkan (hari kalender):")
    print(backtest["macro_age_days"].describe().round(2).to_string())

    print("\nDistribusi umur skor:")
    print(
        backtest["macro_age_days"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    age_report = (
        backtest.groupby("macro_age_days", dropna=False)
        .agg(
            jumlah=("gold_date", "size"),
            skor_tersedia=("macro_score_v2", "count"),
        )
        .reset_index()
    )
    age_report.to_csv(OUTPUT_DIR / "macro_age_audit.csv", index=False)

    # Simpan pasangan tanggal untuk pemeriksaan manual.
    date_cols = [
        c for c in [
            "gold_date",
            "macro_date",
            "macro_age_days",
            "macro_score_v2",
            "macro_bias_v2",
            "available_factors",
            "dollar_observation_date",
            "dollar_age_days",
        ]
        if c in backtest.columns
    ]

    backtest[date_cols].to_csv(
        OUTPUT_DIR / "backtest_timing_pairs.csv", index=False
    )

    print("\nFile audit disimpan:")
    print(OUTPUT_DIR / "gold_weekend_bars.csv")
    print(OUTPUT_DIR / "macro_age_audit.csv")
    print(OUTPUT_DIR / "backtest_timing_pairs.csv")


if __name__ == "__main__":
    main()
