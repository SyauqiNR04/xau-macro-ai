
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    ROOT / "data" / "raw" / "bls_cpi_observations.csv"
)

BLS_API_URL = (
    "https://api.bls.gov/publicAPI/v2/timeseries/data/"
)

# CPI-U All Items, tidak disesuaikan secara musiman.
# Seri ini sesuai untuk dibandingkan dengan CPIAUCNS di FRED.
SERIES_ID = "CUUR0000SA0"

START_YEAR = 2010

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("BLS_API_KEY")


# ============================================================
# MENGAMBIL DATA CPI DARI BLS API
# ============================================================

def fetch_bls_data(start_year, end_year):
    """Mengambil data CPI bulanan dari BLS API."""

    if not API_KEY:
        raise RuntimeError(
            "BLS_API_KEY tidak ditemukan. "
            "Pastikan file .env memiliki BLS_API_KEY."
        )

    payload = {
        "seriesid": [SERIES_ID],
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": API_KEY,
    }

    print(
        f"Mengambil data CPI BLS "
        f"periode {start_year}-{end_year}..."
    )

    response = requests.post(
        BLS_API_URL,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    result = response.json()

    if result.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(
            f"BLS API gagal: {result.get('message', result)}"
        )

    series_list = (
        result.get("Results", {}).get("series", [])
    )

    if not series_list:
        raise RuntimeError(
            "BLS API tidak mengembalikan data CPI."
        )

    rows = []

    for item in series_list[0].get("data", []):
        period = item.get("period", "")

        # Ambil data bulanan M01-M12.
        # Abaikan M13 karena merupakan annual average.
        if period not in [
            f"M{month:02d}" for month in range(1, 13)
        ]:
            continue

        try:
            year = int(item["year"])
            month = int(period[1:])
            value = float(item["value"])

        except (ValueError, TypeError, KeyError):
            # Lewati nilai kosong atau tidak valid.
            continue

        rows.append({
            "series_id": SERIES_ID,
            "observation_date": (
                f"{year}-{month:02d}-01"
            ),
            "value": value,
            "period_name": item.get("periodName", ""),
            "latest": item.get("latest", "false"),
            "source": "BLS",
        })

    if not rows:
        raise RuntimeError(
            "Tidak ada observasi CPI bulanan yang valid."
        )

    df = pd.DataFrame(rows)

    df["observation_date"] = pd.to_datetime(
        df["observation_date"],
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["observation_date", "value"]
    )

    df = df.drop_duplicates(
        subset=["series_id", "observation_date"],
        keep="first",
    )

    df = df.sort_values("observation_date")

    return df.reset_index(drop=True)


# ============================================================
# MEMERIKSA APAKAH DATA PERLU DIPERBARUI
# ============================================================

def latest_data_changed(new_data):
    """
    Membandingkan tanggal dan nilai observasi terbaru
    dengan file lokal.

    Catatan: pemeriksaan ini tidak mendeteksi revisi
    pada bulan-bulan sebelumnya jika data terbaru sama.
    """

    if not OUTPUT_FILE.exists():
        print("File CPI lokal belum tersedia.")
        return True

    try:
        local = pd.read_csv(OUTPUT_FILE)

        required_columns = {
            "series_id",
            "observation_date",
            "value",
        }

        if not required_columns.issubset(local.columns):
            print(
                "Kolom file lokal tidak lengkap. "
                "Data akan dibuat ulang."
            )
            return True

        local["observation_date"] = pd.to_datetime(
            local["observation_date"],
            errors="coerce",
        )

        local["value"] = pd.to_numeric(
            local["value"],
            errors="coerce",
        )

        local = local.dropna(
            subset=["observation_date", "value"]
        )

        if local.empty:
            print("File lokal belum memiliki data valid.")
            return True

        old_latest = (
            local.sort_values("observation_date").iloc[-1]
        )

        new_latest = (
            new_data.sort_values("observation_date").iloc[-1]
        )

        same_date = (
            old_latest["observation_date"]
            == new_latest["observation_date"]
        )

        same_value = (
            abs(
                float(old_latest["value"])
                - float(new_latest["value"])
            )
            < 1e-9
        )

        if same_date and same_value:
            print(
                "Data CPI terbaru belum berubah: "
                f"{new_latest['observation_date'].date()}, "
                f"nilai {new_latest['value']:.3f}."
            )
            return False

        print("Data CPI terbaru berubah.")
        return True

    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"Gagal membaca file lokal: {exc}")
        print("Data akan dibuat ulang.")
        return True


# ============================================================
# PROSES PEMBARUAN DATA
# ============================================================

def main():
    current_year = datetime.now(timezone.utc).year

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Ambil periode terbaru untuk pemeriksaan awal.
    recent_data = fetch_bls_data(
        max(START_YEAR, current_year - 1),
        current_year,
    )

    if not latest_data_changed(recent_data):
        print("Tidak ada pembaruan yang diperlukan.")
        return

    # Jika berubah, ambil ulang seluruh riwayat sejak 2010.
    # Ini juga memungkinkan revisi historis ikut diperbarui.
    historical_data = fetch_bls_data(
        START_YEAR,
        current_year,
    )

    historical_data["retrieved_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    historical_data.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    latest = historical_data.iloc[-1]

    print("\nPembaruan CPI BLS berhasil.")
    print(f"Jumlah observasi: {len(historical_data)}")
    print(f"File tersimpan: {OUTPUT_FILE}")
    print(
        f"Observasi terbaru: "
        f"{latest['observation_date'].date()}"
    )
    print(f"Nilai CPI terbaru: {latest['value']:.3f}")


# ============================================================
# TITIK MASUK PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
