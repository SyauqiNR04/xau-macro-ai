
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "fred_observations.csv"

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("FRED_API_KEY", "").strip()
API_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "DGS10": "US 10-Year Treasury Nominal Yield",
    "DFII10": "US 10-Year Treasury Real Yield",
    "CPIAUCSL": "US Consumer Price Index",
    "CPIAUCNS": "US CPI, Not Seasonally Adjusted",
    "DTWEXBGS": "US Nominal Broad Dollar Index",
}


def fetch_series(series_id, description):
    params = {
        "api_key": API_KEY,
        "series_id": series_id,
        "file_type": "json",
        "observation_start": "2010-01-01",
        "sort_order": "asc",
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "observations" not in payload:
        raise ValueError(f"Data observasi tidak ditemukan: {series_id}")

    df = pd.DataFrame(payload["observations"])

    if df.empty:
        raise ValueError(f"Data kosong: {series_id}")

    df = df.rename(columns={
        "date": "observation_date",
        "value": "value",
    })

    df["series_id"] = series_id
    df["description"] = description
    df["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()

    df["observation_date"] = pd.to_datetime(
        df["observation_date"], errors="coerce"
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df[
        [
            "series_id",
            "description",
            "observation_date",
            "value",
            "realtime_start",
            "realtime_end",
            "retrieved_at_utc",
        ]
    ]


def main():
    if not API_KEY:
        raise RuntimeError(
            "FRED_API_KEY tidak ditemukan. Periksa file .env."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for series_id, description in SERIES.items():
        print(f"Mengambil {series_id} ...")
        df = fetch_series(series_id, description)

        valid_count = int(df["value"].notna().sum())
        print(f"  Total observasi: {len(df)}")
        print(f"  Nilai valid: {valid_count}")

        frames.append(df)

    new_data = pd.concat(frames, ignore_index=True)

    if OUTPUT_FILE.exists():
        old_data = pd.read_csv(
            OUTPUT_FILE,
            parse_dates=["observation_date"],
        )
        combined = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined = new_data

    combined = combined.drop_duplicates(
        subset=[
            "series_id",
            "observation_date",
            "value",
            "realtime_start",
            "realtime_end",
        ],
        keep="last",
    )

    combined = combined.sort_values(
        ["series_id", "observation_date", "retrieved_at_utc"]
    )

    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nPengambilan data selesai.")
    print(f"Lokasi file: {OUTPUT_FILE}")
    print(f"Total baris: {len(combined)}")


if __name__ == "__main__":
    main()
