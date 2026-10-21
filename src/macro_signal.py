
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CPI_FILE = DATA_DIR / "cpi_release_aligned.csv"
YIELD_FILE = DATA_DIR / "yield_features.csv"
OUTPUT_FILE = DATA_DIR / "macro_signal.csv"

CPI_SERIES_ID = "CPIAUCNS"
VALID_CPI_STATUSES = {"released", "actual", "published"}


def load_data():
    """Membaca data CPI dan yield yang telah diproses."""
    for path in (CPI_FILE, YIELD_FILE):
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    cpi = pd.read_csv(CPI_FILE)
    yields = pd.read_csv(YIELD_FILE)

    required_cpi = {
        "reference_month",
        "series_id",
        "cpi_yoy_pct",
        "available_at_utc",
        "release_time_verified",
        "status",
    }
    missing_cpi = required_cpi - set(cpi.columns)

    if missing_cpi:
        raise ValueError(
            f"Kolom CPI belum tersedia: {sorted(missing_cpi)}"
        )

    required_yield = {
        "observation_date",
        "real_yield_10y",
        "real_yield_10y_change_5obs_pp",
    }
    missing_yield = required_yield - set(yields.columns)

    if missing_yield:
        raise ValueError(
            f"Kolom yield belum tersedia: {sorted(missing_yield)}"
        )

    cpi["reference_month"] = pd.to_datetime(
        cpi["reference_month"], errors="coerce"
    )
    cpi["available_at_utc"] = pd.to_datetime(
        cpi["available_at_utc"], utc=True, errors="coerce"
    )

    yields["observation_date"] = pd.to_datetime(
        yields["observation_date"], errors="coerce"
    )

    if cpi["reference_month"].isna().any():
        raise ValueError("Ada reference_month CPI yang tidak valid.")

    if yields["observation_date"].isna().any():
        raise ValueError("Ada observation_date yield yang tidak valid.")

    return cpi, yields


def prepare_cpi(cpi):
    """Memilih CPI NSA dengan waktu rilis yang ditandai terverifikasi."""
    verified = (
        cpi["release_time_verified"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )

    published = (
        cpi["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(VALID_CPI_STATUSES)
    )

    result = cpi.loc[
        (cpi["series_id"] == CPI_SERIES_ID)
        & verified
        & published
    ].copy()

    result = result.dropna(
        subset=["available_at_utc", "cpi_yoy_pct"]
    )

    result = result.sort_values("available_at_utc")

    if result["available_at_utc"].duplicated().any():
        raise ValueError(
            "Ada timestamp rilis CPI duplikat pada seri CPIAUCNS."
        )

    result = result[
        ["available_at_utc", "reference_month", "cpi_yoy_pct"]
    ].rename(
        columns={
            "reference_month": "cpi_reference_month",
            "cpi_yoy_pct": "cpi_yoy_pct_nsa",
        }
    )

    return result


def prepare_yields(yields):
    """
    Membuat timestamp akhir hari New York sebagai asumsi konservatif.

    Ini bukan timestamp publikasi resmi FRED. Untuk backtest intraday,
    timestamp ini belum cukup akurat.
    """
    result = yields.copy()

    result = result.sort_values("observation_date")
    result = result.drop_duplicates(
        subset=["observation_date"], keep="last"
    )

    # Perlakukan tanggal observasi sebagai tanggal kalender New York.
    date_text = result["observation_date"].dt.strftime("%Y-%m-%d")

    end_of_day_et = pd.to_datetime(
        date_text + " 23:59:59",
        errors="raise",
    )

    result["yield_available_at_utc"] = (
        end_of_day_et
        .dt.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="NaT",
        )
        .dt.tz_convert("UTC")
    )

    result["yield_timestamp_basis"] = (
        "estimated_end_of_day_new_york"
    )

    if result["yield_available_at_utc"].isna().any():
        raise ValueError(
            "Ada timestamp yield yang gagal dikonversi ke UTC."
        )

    return result.sort_values("yield_available_at_utc")


def build_macro_signal(cpi, yields):
    """Menggabungkan yield dengan CPI terakhir yang telah tersedia."""
    cpi = prepare_cpi(cpi)
    yields = prepare_yields(yields)

    if cpi.empty:
        raise ValueError(
            "Tidak ada CPI yang lolos validasi timestamp dan status."
        )

    result = pd.merge_asof(
        yields,
        cpi,
        left_on="yield_available_at_utc",
        right_on="available_at_utc",
        direction="backward",
        allow_exact_matches=True,
    )

    # Validasi: CPI tidak boleh berasal dari masa depan.
    has_cpi = result["available_at_utc"].notna()

    invalid_time = (
        has_cpi
        & (
            result["available_at_utc"]
            > result["yield_available_at_utc"]
        )
    )

    if invalid_time.any():
        raise AssertionError(
            "Ditemukan CPI yang timestamp-nya melewati waktu yield."
        )

    # Sinyal arah real yield:
    # yield turun -> tekanan cenderung mendukung emas.
    # yield naik -> tekanan cenderung menekan emas.
    change = pd.to_numeric(
        result["real_yield_10y_change_5obs_pp"],
        errors="coerce",
    )

    result["real_yield_signal"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    result.loc[change < 0, "real_yield_signal"] = 1
    result.loc[change > 0, "real_yield_signal"] = -1
    result.loc[change == 0, "real_yield_signal"] = 0

    result["real_yield_interpretation"] = "Data belum cukup"

    result.loc[
        result["real_yield_signal"] == 1,
        "real_yield_interpretation",
    ] = "Cenderung mendukung emas"

    result.loc[
        result["real_yield_signal"] == -1,
        "real_yield_interpretation",
    ] = "Cenderung menekan emas"

    result.loc[
        result["real_yield_signal"] == 0,
        "real_yield_interpretation",
    ] = "Perubahan real yield netral"

    result["cpi_available"] = has_cpi
    result["cpi_timestamp_valid"] = (
        result["available_at_utc"].isna()
        | (
            result["available_at_utc"]
            <= result["yield_available_at_utc"]
        )
    )

    result = result.sort_values("observation_date")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    return result


def main():
    cpi, yields = load_data()
    signals = build_macro_signal(cpi, yields)

    print("Macro signal selesai dibuat.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Jumlah baris: {len(signals)}")
    print(
        "Baris dengan CPI tersedia:",
        int(signals["cpi_available"].sum()),
    )
    print(
        "Baris tanpa CPI:",
        int((~signals["cpi_available"]).sum()),
    )
    print(
        "Timestamp CPI valid:",
        bool(signals["cpi_timestamp_valid"].all()),
    )

    print("\nDistribusi sinyal real yield:")
    print(
        signals["real_yield_signal"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    columns = [
        "observation_date",
        "yield_available_at_utc",
        "real_yield_10y",
        "real_yield_10y_change_5obs_pp",
        "real_yield_signal",
        "real_yield_interpretation",
        "cpi_reference_month",
        "cpi_yoy_pct_nsa",
        "available_at_utc",
        "yield_timestamp_basis",
    ]

    print("\nSinyal terbaru:")
    print(signals[columns].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
