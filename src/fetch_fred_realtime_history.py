import os
import re
import time
from pathlib import Path

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

# Gunakan snapshot tanggal 1 setiap bulan.
# Setiap snapshot merepresentasikan data yang tercatat
# pada vintage date tersebut, bukan tanggal rilis tiap observasi.
vintage_dates = pd.date_range(
    start=START,
    end=END,
    freq="MS",
).strftime("%Y-%m-%d").tolist()

# Tambahkan snapshot terbaru untuk periode terakhir.
if END not in vintage_dates:
    vintage_dates.append(END)

SERIES = {
    "DGS10": "Nominal Treasury yield 10Y",
    "DFII10": "Real Treasury yield 10Y",
    "CPIAUCNS": "CPI All Urban Consumers NSA",
}

URL = "https://api.stlouisfed.org/fred/series/observations"
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
all_results = []

for series_id, description in SERIES.items():
    print(f"\nMengambil vintage {series_id}...", flush=True)

    params = {
        "api_key": API_KEY,
        "series_id": series_id,
        "file_type": "json",
        "observation_start": START,
        "observation_end": END,
        "vintage_dates": ",".join(vintage_dates),
        "output_type": 2,
        "limit": 100000,
        "offset": 0,
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
            f"FRED gagal untuk {series_id}. "
            f"HTTP {response.status_code}: {error}"
        )

    payload = response.json()
    observations = payload.get("observations", [])

    if not observations:
        print(f"Tidak ada observasi untuk {series_id}.")
        continue

    df = pd.DataFrame(observations)

    # Cari kolom nilai per tanggal vintage, contohnya DGS10_20230101.
    vintage_columns = [
        col for col in df.columns
        if re.fullmatch(
            rf"{re.escape(series_id)}_\d{{8}}", col
        )
    ]

    if not vintage_columns:
        raise RuntimeError(
            f"Kolom vintage tidak ditemukan untuk {series_id}. "
            f"Kolom respons: {list(df.columns)[:20]}"
        )

    if "date" not in df.columns:
        raise RuntimeError(
            f"Kolom date tidak ditemukan untuk {series_id}."
        )

    long_df = df[["date"] + vintage_columns].melt(
        id_vars=["date"],
        var_name="vintage_column",
        value_name="value",
    )

    long_df["vintage_date"] = (
        long_df["vintage_column"]
        .str.extract(r"_(\d{8})$")[0]
        .pipe(lambda s: pd.to_datetime(s, format="%Y%m%d"))
    )

    long_df["observation_date"] = pd.to_datetime(
        long_df["date"], errors="coerce"
    )
    long_df["value"] = pd.to_numeric(
        long_df["value"], errors="coerce"
    )
    long_df["series_id"] = series_id
    long_df["description"] = description

    long_df = long_df.rename(
        columns={"value": "vintage_value"}
    )

    long_df = long_df[
        [
            "series_id",
            "description",
            "observation_date",
            "vintage_date",
            "vintage_value",
        ]
    ]

    long_df = long_df.dropna(
        subset=["observation_date", "vintage_date"]
    )

    # Simpan per seri agar hasil yang sudah berhasil tidak hilang.
    series_path = OUT_DIR / f"fred_vintage_{series_id}.csv"
    long_df.to_csv(series_path, index=False)

    all_results.append(long_df)

    print("Snapshot diminta:", len(vintage_dates))
    print("Kolom vintage diterima:", len(vintage_columns))
    print("Baris hasil:", len(long_df))
    print("File:", series_path)
    time.sleep(0.25)

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.sort_values(
        ["series_id", "vintage_date", "observation_date"]
    )

    combined_path = OUT_DIR / "fred_vintage_macro.csv"
    combined.to_csv(combined_path, index=False)

    print("\n=== SELESAI ===")
    print("File gabungan:", combined_path)
    print("Total baris:", len(combined))
    print("\nJumlah baris per seri:")
    print(combined.groupby("series_id").size().to_string())

    print("\nContoh data:")
    print(combined.head(10).to_string(index=False))

print(
    "\nDTWEXBGS sengaja belum dimasukkan: "
    "FRED melaporkan seri tersebut tidak tersedia di ALFRED "
    "untuk permintaan vintage ini. Jangan gunakan snapshot terbaru "
    "sebagai pengganti data historis point-in-time."
)
