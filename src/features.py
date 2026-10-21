
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "fred_observations.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["observation_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["observation_date", "value"])

    return df



def build_cpi_features(df):
    outputs = []

    series_info = {
        "CPIAUCNS": "not_seasonally_adjusted",
        "CPIAUCSL": "seasonally_adjusted",
    }

    for series_id, adjustment in series_info.items():
        cpi = df.loc[
            df["series_id"] == series_id,
            ["observation_date", "value"],
        ].copy()

        if cpi.empty:
            print(f"Peringatan: seri {series_id} tidak ditemukan.")
            continue

        cpi["month"] = cpi["observation_date"].dt.to_period("M")
        cpi = cpi.sort_values("observation_date")
        cpi = cpi.drop_duplicates("month", keep="last")

        # Gunakan indeks bulan kalender agar bulan yang hilang
        # tidak membuat periode pembanding bergeser.
        cpi = cpi.set_index("month").sort_index()
        full_months = pd.period_range(
            start=cpi.index.min(),
            end=cpi.index.max(),
            freq="M",
        )
        cpi = cpi.reindex(full_months)
        cpi.index.name = "month"

        cpi["cpi_yoy_pct"] = (
            cpi["value"].pct_change(
                periods=12,
                fill_method=None,
            ) * 100
        )

        cpi["observation_date"] = cpi.index.to_timestamp()
        cpi = cpi.rename(columns={"value": "cpi_index"})
        cpi["series_id"] = series_id
        cpi["adjustment"] = adjustment

        outputs.append(cpi.reset_index(drop=True))

    if not outputs:
        raise ValueError("Tidak ada seri CPI yang berhasil diproses.")

    result = pd.concat(outputs, ignore_index=True)

    return result[
        [
            "observation_date",
            "series_id",
            "adjustment",
            "cpi_index",
            "cpi_yoy_pct",
        ]
    ].sort_values(["series_id", "observation_date"])



def build_yield_features(df):
    outputs = []

    for series_id, prefix in [
        ("DGS10", "nominal_yield_10y"),
        ("DFII10", "real_yield_10y"),
    ]:
        part = df.loc[
            df["series_id"] == series_id,
            ["observation_date", "value"],
        ].copy()

        if part.empty:
            raise ValueError(f"Seri yield {series_id} tidak ditemukan.")

        part = part.sort_values("observation_date")
        part = part.drop_duplicates("observation_date", keep="last")
        part = part.rename(columns={"value": prefix})

        # Perubahan berdasarkan jumlah observasi yang tersedia.
        part[f"{prefix}_change_1obs_pp"] = part[prefix].diff(1)
        part[f"{prefix}_change_5obs_pp"] = part[prefix].diff(5)

        outputs.append(part)

    nominal = outputs[0]
    real = outputs[1]

    yields = pd.merge(
        nominal,
        real,
        on="observation_date",
        how="outer",
    ).sort_values("observation_date")

    # Nilai positif berarti real yield turun dalam 5 observasi.
    # Ini merupakan proksi arah tekanan, bukan sinyal trading.
    yields["real_yield_pressure_for_gold"] = (
        -yields["real_yield_10y_change_5obs_pp"]
    )

    return yields


def print_latest_cpi(cpi_features, rows=3):
    for series_id in ["CPIAUCNS", "CPIAUCSL"]:
        latest = (
            cpi_features.loc[cpi_features["series_id"] == series_id]
            .sort_values("observation_date")
            .tail(rows)
        )

        print(f"\nCPI terbaru — {series_id}:")

        if latest.empty:
            print("Tidak ada data untuk seri ini.")
        else:
            print(latest.to_string(index=False))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    cpi_features = build_cpi_features(raw)
    yield_features = build_yield_features(raw)

    cpi_file = OUTPUT_DIR / "cpi_features.csv"
    yield_file = OUTPUT_DIR / "yield_features.csv"

    cpi_features.to_csv(cpi_file, index=False)
    yield_features.to_csv(yield_file, index=False)

    print("Feature engineering selesai.")
    print(f"Data CPI: {cpi_file}")
    print(f"Jumlah observasi CPI: {len(cpi_features)}")
    print(f"Data yield: {yield_file}")
    print(f"Jumlah observasi yield: {len(yield_features)}")

    print_latest_cpi(cpi_features)

    print("\nYield terbaru:")
    print(yield_features.tail(5).to_string(index=False))

    print("\n=== VALIDASI DATA ===")

    print("\nCPI:")
    print(
        "Rentang:",
        cpi_features["observation_date"].min(),
        "sampai",
        cpi_features["observation_date"].max(),
    )

    for series_id in ["CPIAUCNS", "CPIAUCSL"]:
        subset = cpi_features[
            cpi_features["series_id"] == series_id
        ]

        print(f"\n{series_id}:")
        print(f"Jumlah baris: {len(subset)}")
        print(f"CPI YoY kosong: {subset['cpi_yoy_pct'].isna().sum()}")

        valid = subset.dropna(subset=["cpi_yoy_pct"])
        print("Tiga observasi YoY terakhir:")

        if valid.empty:
            print("Belum ada data YoY yang valid.")
        else:
            print(valid.tail(3).to_string(index=False))

    print("\nYIELD:")
    print(
        "Rentang:",
        yield_features["observation_date"].min(),
        "sampai",
        yield_features["observation_date"].max(),
    )
    print("Nilai real yield kosong:",
          yield_features["real_yield_10y"].isna().sum())
    print(yield_features.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
