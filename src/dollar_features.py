
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "fred_observations.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dollar_features.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["observation_date"],
    )

    dollar = df.loc[
        df["series_id"] == "DTWEXBGS",
        ["observation_date", "value", "retrieved_at_utc"],
    ].copy()

    if dollar.empty:
        raise ValueError(
            "Seri DTWEXBGS tidak ditemukan. Jalankan data_ingestion.py dahulu."
        )

    dollar["value"] = pd.to_numeric(dollar["value"], errors="coerce")
    dollar = dollar.dropna(subset=["observation_date", "value"])

    # Ambil satu nilai terakhir untuk setiap tanggal observasi.
    dollar = dollar.sort_values(
        ["observation_date", "retrieved_at_utc"]
    )
    dollar = dollar.drop_duplicates(
        subset=["observation_date"],
        keep="last",
    )
    dollar = dollar.sort_values("observation_date").reset_index(drop=True)

    # Perubahan indeks dalam persen dibanding observasi sebelumnya.
    dollar["dollar_change_1obs_pct"] = (
        dollar["value"].pct_change(fill_method=None) * 100
    )
    dollar["dollar_change_5obs_pct"] = (
        dollar["value"].pct_change(periods=5, fill_method=None) * 100
    )

    # Kenaikan dolar cenderung menjadi hambatan bagi emas.
    dollar["dollar_signal"] = 0
    dollar.loc[
        dollar["dollar_change_5obs_pct"] < 0,
        "dollar_signal",
    ] = 1
    dollar.loc[
        dollar["dollar_change_5obs_pct"] > 0,
        "dollar_signal",
    ] = -1

    dollar.loc[
        dollar["dollar_change_5obs_pct"].isna(),
        "dollar_signal",
    ] = pd.NA

    dollar["dollar_interpretation"] = "Perubahan dolar netral"
    dollar.loc[
        dollar["dollar_signal"] == 1,
        "dollar_interpretation",
    ] = "Dolar melemah: cenderung mendukung emas"
    dollar.loc[
        dollar["dollar_signal"] == -1,
        "dollar_interpretation",
    ] = "Dolar menguat: cenderung menekan emas"
    dollar.loc[
        dollar["dollar_signal"].isna(),
        "dollar_interpretation",
    ] = "Data belum cukup"

    dollar = dollar.rename(columns={"value": "dollar_index"})

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dollar.to_csv(OUTPUT_FILE, index=False)

    print("\n=== DOLLAR FEATURES SELESAI ===")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Jumlah observasi: {len(dollar)}")
    print(f"Nilai indeks valid: {dollar['dollar_index'].notna().sum()}")
    print("\nDistribusi sinyal dolar:")
    print(dollar["dollar_signal"].value_counts(dropna=False).sort_index())
    print("\n10 observasi terbaru:")
    print(
        dollar[
            [
                "observation_date",
                "dollar_index",
                "dollar_change_1obs_pct",
                "dollar_change_5obs_pct",
                "dollar_signal",
                "dollar_interpretation",
            ]
        ].tail(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
