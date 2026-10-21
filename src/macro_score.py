
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "macro_signal.csv"
OUTPUT_FILE = DATA_DIR / "macro_score.csv"


def load_data():
    """Memuat sinyal makro yang telah dibuat sebelumnya."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "observation_date",
        "real_yield_10y_change_5obs_pp",
        "real_yield_signal",
        "cpi_reference_month",
        "cpi_yoy_pct_nsa",
        "available_at_utc",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df["observation_date"] = pd.to_datetime(
        df["observation_date"], errors="coerce"
    )

    df["cpi_reference_month"] = pd.to_datetime(
        df["cpi_reference_month"], errors="coerce"
    )

    df["available_at_utc"] = pd.to_datetime(
        df["available_at_utc"], utc=True, errors="coerce"
    )

    df["real_yield_signal"] = pd.to_numeric(
        df["real_yield_signal"], errors="coerce"
    )

    df["real_yield_10y_change_5obs_pp"] = pd.to_numeric(
        df["real_yield_10y_change_5obs_pp"], errors="coerce"
    )

    df["cpi_yoy_pct_nsa"] = pd.to_numeric(
        df["cpi_yoy_pct_nsa"], errors="coerce"
    )

    df = df.dropna(subset=["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    return df


def add_cpi_context(df):
    """
    Menambahkan konteks perubahan CPI YoY.

    Perubahan CPI hanya menjadi konteks ekonomi, bukan sinyal
    bullish/bearish otomatis, karena ekspektasi pasar belum tersedia.
    """
    df = df.copy()

    # Satu baris per bulan referensi CPI.
    cpi_history = (
        df.dropna(
            subset=["cpi_reference_month", "cpi_yoy_pct_nsa"]
        )[
            ["cpi_reference_month", "cpi_yoy_pct_nsa"]
        ]
        .drop_duplicates(subset=["cpi_reference_month"])
        .sort_values("cpi_reference_month")
        .copy()
    )

    cpi_history["cpi_yoy_change_pp"] = (
        cpi_history["cpi_yoy_pct_nsa"].diff()
    )

    cpi_history["cpi_trend_context"] = "Data CPI belum cukup"

    cpi_history.loc[
        cpi_history["cpi_yoy_change_pp"] > 0,
        "cpi_trend_context",
    ] = "Inflasi tahunan meningkat"

    cpi_history.loc[
        cpi_history["cpi_yoy_change_pp"] < 0,
        "cpi_trend_context",
    ] = "Inflasi tahunan menurun"

    cpi_history.loc[
        cpi_history["cpi_yoy_change_pp"] == 0,
        "cpi_trend_context",
    ] = "Inflasi tahunan tidak berubah"

    df = df.merge(
        cpi_history[
            [
                "cpi_reference_month",
                "cpi_yoy_change_pp",
                "cpi_trend_context",
            ]
        ],
        on="cpi_reference_month",
        how="left",
        validate="many_to_one",
    )

    return df


def calculate_macro_score(df):
    """
    Skor versi 1: menggunakan arah perubahan real yield.

    +100 = real yield turun, cenderung mendukung emas.
    0    = real yield tidak berubah.
    -100 = real yield naik, cenderung menekan emas.
    NaN  = data real yield belum cukup.

    Skor ini adalah indikator arah, bukan probabilitas harga.
    """
    df = df.copy()

    signal = df["real_yield_signal"]

    valid_signal = signal.isin([-1, 0, 1])

    df["real_yield_score"] = (
        100 * signal
    ).where(valid_signal)

    df["macro_score"] = df["real_yield_score"]

    df["macro_bias"] = "Data belum cukup"

    df.loc[
        df["macro_score"] >= 50,
        "macro_bias",
    ] = "Cenderung bullish emas"

    df.loc[
        df["macro_score"] <= -50,
        "macro_bias",
    ] = "Cenderung bearish emas"

    df.loc[
        df["macro_score"].between(-49.999, 49.999),
        "macro_bias",
    ] = "Netral"

    df["score_coverage_pct"] = (
        df["real_yield_score"].notna().astype(int) * 100
    )

    df["score_method"] = "real_yield_direction_v1"

    df["score_warning"] = (
        "Indikator awal; bukan probabilitas atau rekomendasi entry"
    )

    return df


def main():
    df = load_data()
    df = add_cpi_context(df)
    df = calculate_macro_score(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("=== MACRO SCORE V1 SELESAI ===")
    print(f"File output: {OUTPUT_FILE}")
    print(f"Jumlah baris: {len(df)}")

    print("\nDistribusi bias makro:")
    print(df["macro_bias"].value_counts(dropna=False).to_string())

    print("\nKetersediaan skor:")
    print(
        df["macro_score"]
        .notna()
        .value_counts()
        .rename(index={True: "Skor tersedia", False: "Skor belum tersedia"})
        .to_string()
    )

    print("\nKonteks tren CPI:")
    print(df["cpi_trend_context"].value_counts(dropna=False).to_string())

    columns = [
        "observation_date",
        "real_yield_10y_change_5obs_pp",
        "real_yield_score",
        "cpi_reference_month",
        "cpi_yoy_pct_nsa",
        "cpi_yoy_change_pp",
        "cpi_trend_context",
        "macro_score",
        "macro_bias",
    ]

    print("\n10 observasi terbaru:")
    print(df[columns].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
