
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT / "data" / "raw" / "fred_observations.csv"
CALENDAR_FILE = (
    ROOT / "data" / "processed" / "cpi_release_calendar.csv"
)
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "cpi_release_aligned.csv"
)

CPI_SERIES = {
    "CPIAUCNS": "not_seasonally_adjusted",
    "CPIAUCSL": "seasonally_adjusted",
}

RELEASED_STATUSES = {
    "released",
    "actual",
    "published",
    "released_archive_found",
}


def parse_bool_column(series):
    """Mengubah nilai boolean dari CSV menjadi True/False."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def load_inputs():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {RAW_FILE}")

    if not CALENDAR_FILE.exists():
        raise FileNotFoundError(
            f"Kalender rilis tidak ditemukan: {CALENDAR_FILE}"
        )

    raw = pd.read_csv(RAW_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)

    required_raw = {
        "series_id",
        "observation_date",
        "value",
    }
    missing_raw = required_raw - set(raw.columns)

    if missing_raw:
        raise ValueError(
            f"Kolom data FRED tidak tersedia: {sorted(missing_raw)}"
        )

    if "reference_month" not in calendar.columns:
        raise ValueError(
            "Kalender tidak memiliki kolom reference_month."
        )

    # Kompatibilitas dengan format kalender mentah yang lama.
    if (
        "actual_release_date" not in calendar.columns
        and "release_date" in calendar.columns
    ):
        calendar = calendar.rename(
            columns={"release_date": "actual_release_date"}
        )

    if (
        "release_datetime_utc" in calendar.columns
        and "release_time_et" not in calendar.columns
    ):
        calendar["release_time_et"] = pd.NA

    if "release_time_et" not in calendar.columns:
        calendar["release_time_et"] = pd.NA

    if "status" not in calendar.columns:
        calendar["status"] = "unverified"

    if "source" not in calendar.columns:
        if "source_url" in calendar.columns:
            calendar["source"] = calendar["source_url"]
        else:
            calendar["source"] = pd.NA

    raw["observation_date"] = pd.to_datetime(
        raw["observation_date"], errors="coerce"
    )
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

    calendar["reference_month"] = (
        pd.to_datetime(
            calendar["reference_month"], errors="coerce"
        )
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    calendar["actual_release_date"] = pd.to_datetime(
        calendar["actual_release_date"], errors="coerce"
    ).dt.normalize()

    calendar["status"] = (
        calendar["status"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Samakan status kalender mentah dengan status pipeline.
    calendar["status"] = calendar["status"].replace({
        "released_archive_found": "published",
        "not_published": "cancelled",
    })

    if calendar["reference_month"].isna().any():
        raise ValueError(
            "Ada reference_month kalender yang tidak valid."
        )

    if calendar["reference_month"].duplicated().any():
        raise ValueError(
            "Kalender memiliki reference_month duplikat."
        )

    # Tanggal yang ditemukan pada arsip resmi dibedakan dari
    # waktu rilis yang benar-benar sudah diverifikasi.
    calendar["release_date_verified"] = (
        calendar["actual_release_date"].notna()
        & calendar["status"].isin(RELEASED_STATUSES)
    )

    # Hanya metadata verifikasi eksplisit yang dianggap valid.
    # Catatan standar 08:30 ET saja bukan bukti verifikasi waktu.
    if "release_time_verified" in calendar.columns:
        calendar["release_time_verified"] = parse_bool_column(
            calendar["release_time_verified"]
        )
    else:
        calendar["release_time_verified"] = False

    return raw, calendar


def build_monthly_cpi(raw, series_id, adjustment):
    part = raw.loc[
        raw["series_id"] == series_id,
        ["observation_date", "value"],
    ].copy()

    if part.empty:
        raise ValueError(
            f"Seri {series_id} tidak ditemukan di data FRED."
        )

    part["reference_month"] = (
        part["observation_date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    part = part.sort_values("observation_date")
    part = part.drop_duplicates(
        "reference_month", keep="last"
    )

    part = part[
        ["reference_month", "value"]
    ].set_index("reference_month")

    # Pertahankan bulan yang hilang sebagai NaN.
    full_months = pd.date_range(
        start=part.index.min(),
        end=part.index.max(),
        freq="MS",
        name="reference_month",
    )

    part = part.reindex(full_months)
    part.index.name = "reference_month"
    part = part.reset_index()

    part["cpi_yoy_pct"] = (
        part["value"].pct_change(
            periods=12, fill_method=None
        ) * 100
    )

    part["series_id"] = series_id
    part["adjustment"] = adjustment

    return part.rename(columns={"value": "cpi_index"})


def build_release_timestamps(calendar):
    result = calendar.copy()

    date_text = result["actual_release_date"].dt.strftime(
        "%Y-%m-%d"
    )
    time_text = (
        result["release_time_et"]
        .astype("string")
        .str.strip()
    )

    valid_status = result["status"].isin(RELEASED_STATUSES)
    valid_date = result["actual_release_date"].notna()
    valid_time = (
        time_text.notna()
        & time_text.ne("")
        & time_text.ne("<NA>")
    )

    release_text = date_text + " " + time_text
    parsed = pd.to_datetime(release_text, errors="coerce")

    valid_release = (
        valid_status
        & valid_date
        & valid_time
        & parsed.notna()
    )

    result["available_at_utc"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )

    if valid_release.any():
        localized_et = parsed.loc[valid_release].dt.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="NaT",
        )

        result.loc[valid_release, "available_at_utc"] = (
            localized_et.dt.tz_convert("UTC")
        )

    result["available_at_et"] = (
        result["available_at_utc"]
        .dt.tz_convert("America/New_York")
    )

    # Tanggal arsip dapat diketahui walaupun waktu persisnya
    # masih menggunakan asumsi pukul 08:30 ET.
    result["timestamp_basis"] = "unavailable"

    has_timestamp = result["available_at_utc"].notna()

    result.loc[
        has_timestamp, "timestamp_basis"
    ] = "estimated_standard_release_time"

    verified = (
        has_timestamp
        & result["release_time_verified"]
    )

    result.loc[
        verified, "timestamp_basis"
    ] = "verified_release_time"

    return result


def main():
    raw, calendar = load_inputs()
    calendar = build_release_timestamps(calendar)

    outputs = []

    for series_id, adjustment in CPI_SERIES.items():
        cpi = build_monthly_cpi(
            raw, series_id, adjustment
        )

        aligned = cpi.merge(
            calendar,
            on="reference_month",
            how="left",
            validate="one_to_one",
        )

        outputs.append(aligned)

    result = pd.concat(outputs, ignore_index=True)
    result = result.sort_values(
        ["series_id", "reference_month"]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    result.to_csv(OUTPUT_FILE, index=False)

    print("Penyelarasan CPI selesai.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Jumlah baris: {len(result)}")

    print("\n=== VALIDASI KALENDER ===")

    for column in [
        "published",
        "cancelled",
        "unverified",
    ]:
        count = (
            result["status"].eq(column).sum()
            if "status" in result.columns
            else 0
        )
        print(f"Status {column}: {count}")

    print(
        "Tanggal rilis diketahui:",
        int(result["release_date_verified"].fillna(False).sum()),
    )
    print(
        "Waktu rilis terverifikasi:",
        int(result["release_time_verified"].fillna(False).sum()),
    )
    print(
        "Timestamp tersedia:",
        int(result["available_at_utc"].notna().sum()),
    )

    columns = [
        "reference_month",
        "series_id",
        "cpi_index",
        "cpi_yoy_pct",
        "actual_release_date",
        "release_time_et",
        "status",
        "available_at_utc",
        "release_date_verified",
        "release_time_verified",
        "timestamp_basis",
    ]

    for series_id in CPI_SERIES:
        print(f"\nData terbaru: {series_id}")

        latest = (
            result[result["series_id"] == series_id]
            .sort_values("reference_month")
            .tail(5)
        )

        print(
            latest[columns].to_string(index=False)
        )

    print(
        "\nCatatan: timestamp dengan basis "
        "'estimated_standard_release_time' "
        "belum layak dianggap sebagai waktu rilis terverifikasi."
    )


if __name__ == "__main__":
    main()
