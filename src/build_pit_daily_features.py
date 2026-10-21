import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path.cwd()
load_dotenv(ROOT / ".env", override=True)

API_KEY = os.getenv("FRED_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit("FRED_API_KEY tidak ditemukan di .env.")

START = "2015-01-01"
END = "2026-10-20"

SERIES = ["DGS10", "DFII10", "CPIAUCNS"]
URL = "https://api.stlouisfed.org/fred/series/observations"
OUT = ROOT / "data" / "processed" / "fred_pit_daily_features.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

session = requests.Session()
all_features = []

def extract_features(df, series_id, vintage_dates):
    if "date" not in df.columns:
        raise RuntimeError(
            f"Kolom date tidak ditemukan untuk {series_id}."
        )

    obs_dates = pd.to_datetime(df["date"], errors="coerce")
    values = df.drop(columns=["date"]).apply(
        pd.to_numeric, errors="coerce"
    )

    rows = []

    for vintage in vintage_dates:
        col = f"{series_id}_{vintage.strftime('%Y%m%d')}"

        # Beberapa tanggal vintage mungkin tidak dikembalikan ALFRED.
        if col not in values.columns:
            continue

        # Hindari memakai observasi bertanggal sama dengan evaluasi.
        mask = (obs_dates < vintage) & values[col].notna()

        series = pd.Series(
            values.loc[mask, col].to_numpy(),
            index=obs_dates.loc[mask],
        ).sort_index()

        row = {
            "date": vintage,
            "series_id": series_id,
            "latest_observation_date": pd.NaT,
            "latest_value": np.nan,
            "change_1obs": np.nan,
            "change_5obs": np.nan,
            "yoy_pct": np.nan,
        }

        if series.empty:
            rows.append(row)
            continue

        row["latest_observation_date"] = series.index[-1]
        row["latest_value"] = float(series.iloc[-1])

        if series_id in ("DGS10", "DFII10"):
            if len(series) >= 2:
                row["change_1obs"] = float(
                    series.iloc[-1] - series.iloc[-2]
                )
            if len(series) >= 6:
                row["change_5obs"] = float(
                    series.iloc[-1] - series.iloc[-6]
                )

        elif series_id == "CPIAUCNS":
            # Cari pasangan CPI untuk bulan yang sama tahun sebelumnya.
            monthly = series.copy()
            monthly.index = monthly.index.to_period("M")
            monthly = monthly[~monthly.index.duplicated(keep="last")]

            current_month = monthly.index[-1]
            prior_month = current_month - 12

            if prior_month in monthly.index:
                prior_value = monthly.loc[prior_month]
                current_value = monthly.loc[current_month]

                if prior_value != 0:
                    row["yoy_pct"] = float(
                        (current_value / prior_value - 1) * 100
                    )

        rows.append(row)

    return pd.DataFrame(rows)

for year in range(2015, 2027):
    year_start = max(pd.Timestamp(START), pd.Timestamp(f"{year}-01-01"))
    year_end = min(pd.Timestamp(END), pd.Timestamp(f"{year}-12-31"))

    if year_start > year_end:
        continue

    vintages = list(pd.date_range(year_start, year_end, freq="D"))

    print(
        f"\n=== {year}: {len(vintages)} tanggal vintage ===",
        flush=True,
    )

    for series_id in SERIES:
        print(f"Mengambil {series_id}...", flush=True)

        params = {
            "api_key": API_KEY,
            "series_id": series_id,
            "file_type": "json",
            "observation_start": START,
            "observation_end": END,
            "vintage_dates": ",".join(
                d.strftime("%Y-%m-%d") for d in vintages
            ),
            "output_type": 2,
            "limit": 100000,
        }

        response = session.get(URL, params=params, timeout=120)

        if not response.ok:
            try:
                error = response.json().get(
                    "error_message", "Tidak ada detail"
                )
            except ValueError:
                error = response.text[:300]

            raise RuntimeError(
                f"FRED gagal untuk {series_id}, tahun {year}. "
                f"HTTP {response.status_code}: {error}"
            )

        payload = response.json()
        raw = pd.DataFrame(payload.get("observations", []))

        if raw.empty:
            print("  Tidak ada observasi pada respons.")
            continue

        features = extract_features(raw, series_id, vintages)
        all_features.append(features)

        print(
            f"  Fitur diekstrak: {len(features)} baris",
            flush=True,
        )

        time.sleep(0.2)

if not all_features:
    raise SystemExit("Tidak ada fitur yang berhasil dibuat.")

combined = pd.concat(all_features, ignore_index=True)

wide = combined.pivot(
    index="date",
    columns="series_id",
    values=[
        "latest_observation_date",
        "latest_value",
        "change_1obs",
        "change_5obs",
        "yoy_pct",
    ],
)

wide.columns = [
    f"{metric}_{series_id}"
    for metric, series_id in wide.columns
]
wide = wide.reset_index()
wide.columns.name = None

wide.to_csv(OUT, index=False)

print("\n=== HASIL AKHIR ===")
print("File:", OUT)
print("Jumlah tanggal:", len(wide))
print("Jumlah kolom:", len(wide.columns))
print("\nKolom:")
print("\n".join(wide.columns))
print("\nContoh 5 baris terakhir:")
print(wide.tail(5).to_string(index=False))
print("\nNilai kosong per kolom:")
print(wide.isna().sum().to_string())
