
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "processed" / "cpi_release_aligned.csv"


def main():
    df = pd.read_csv(FILE)

    cpi = df.loc[df["series_id"] == "CPIAUCNS"].copy()
    cpi["reference_month"] = pd.to_datetime(cpi["reference_month"])
    cpi["available_at_utc"] = pd.to_datetime(
        cpi["available_at_utc"], utc=True, errors="coerce"
    )

    verified = (
        cpi["release_time_verified"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    print("=== AUDIT KALENDER CPI ===")
    print("Rentang bulan:", cpi["reference_month"].min(),
          "sampai", cpi["reference_month"].max())
    print("Total bulan:", len(cpi))
    print("Waktu rilis terverifikasi:", int(verified.sum()))
    print("Waktu rilis belum terverifikasi:", int((~verified).sum()))

    print("\nStatus kalender:")
    print(cpi["status"].fillna("kosong").value_counts().to_string())

    print("\nBulan dengan waktu rilis belum terverifikasi:")
    print(
        cpi.loc[
            ~verified,
            ["reference_month", "status", "actual_release_date",
             "release_time_et", "available_at_utc"]
        ].tail(15).to_string(index=False)
    )

    print("\nRilis terverifikasi terbaru:")
    print(
        cpi.loc[
            verified,
            ["reference_month", "actual_release_date",
             "release_time_et", "available_at_utc", "status"]
        ].tail(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
