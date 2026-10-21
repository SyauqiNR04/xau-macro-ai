
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "fred_observations.csv"

EXPECTED_SERIES = ["CPIAUCSL", "DGS10", "DFII10"]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df["observation_date"] = pd.to_datetime(
        df["observation_date"], errors="coerce"
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    print("=== AUDIT KUALITAS DATA XAU-MACRO AI ===")
    print(f"Total baris mentah: {len(df)}")
    print(f"Rentang tanggal: {df['observation_date'].min().date()} "
          f"sampai {df['observation_date'].max().date()}")

    print("\n=== 1. PEMERIKSAAN PER SERI ===")

    for series_id in EXPECTED_SERIES:
        part = df[df["series_id"] == series_id].copy()

        if part.empty:
            print(f"\n{series_id}: DATA TIDAK DITEMUKAN")
            continue

        duplicates = part.duplicated(
            subset=["observation_date"], keep=False
        ).sum()

        missing_dates = part["observation_date"].isna().sum()
        missing_values = part["value"].isna().sum()
        valid = part.dropna(subset=["observation_date", "value"])

        print(f"\nSeri: {series_id}")
        print(f"Total baris: {len(part)}")
        print(f"Nilai valid: {len(valid)}")
        print(f"Tanggal kosong/tidak valid: {missing_dates}")
        print(f"Nilai kosong/tidak valid: {missing_values}")
        print(f"Baris dengan tanggal duplikat: {duplicates}")

        if not valid.empty:
            print(f"Tanggal terbaru: "
                  f"{valid['observation_date'].max().date()}")
            print(f"Nilai minimum: {valid['value'].min():.4f}")
            print(f"Nilai maksimum: {valid['value'].max():.4f}")

        if series_id == "CPIAUCSL":
            dates = (
                part.dropna(subset=["observation_date"])
                ["observation_date"]
                .drop_duplicates()
            )

            months = set(dates.dt.to_period("M"))
            if months:
                expected_months = pd.period_range(
                    min(months), max(months), freq="M"
                )
                missing_months = sorted(set(expected_months) - months)

                print(f"Bulan yang tidak memiliki baris: "
                      f"{len(missing_months)}")

                if missing_months:
                    print("Contoh bulan hilang:",
                          ", ".join(str(m) for m in missing_months[:12]))

            invalid_cpi = valid[valid["value"] <= 0]
            print(f"Nilai CPI <= 0: {len(invalid_cpi)}")

        elif series_id in ["DGS10", "DFII10"]:
            invalid_yield = valid[
                (valid["value"] < -10) | (valid["value"] > 25)
            ]
            print(f"Nilai yield di luar rentang pemeriksaan "
                  f"[-10, 25]: {len(invalid_yield)}")

            dates = (
                valid["observation_date"]
                .drop_duplicates()
                .sort_values()
            )
            gaps = dates.diff().dt.days
            long_gaps = gaps[gaps > 7]

            print(f"Jeda antarobservasi lebih dari 7 hari: "
                  f"{len(long_gaps)}")

    print("\n=== 2. PEMERIKSAAN DATA GABUNGAN ===")

    duplicate_count = df.duplicated(
        subset=["series_id", "observation_date"]
    ).sum()

    unknown_series = sorted(
        set(df["series_id"].dropna()) - set(EXPECTED_SERIES)
    )

    print(f"Duplikat seri + tanggal: {duplicate_count}")
    print(f"Jumlah baris dengan tanggal tidak valid: "
          f"{df['observation_date'].isna().sum()}")
    print(f"Seri tak terduga: {unknown_series or 'Tidak ada'}")

    print("\nAudit selesai. Data asli tidak diubah.")


if __name__ == "__main__":
    main()
