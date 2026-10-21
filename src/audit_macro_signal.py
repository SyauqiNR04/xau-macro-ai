
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

SIGNAL_FILE = DATA / "macro_signal.csv"
CPI_FILE = DATA / "cpi_release_aligned.csv"
YIELD_FILE = DATA / "yield_features.csv"


def main():
    signals = pd.read_csv(SIGNAL_FILE)
    cpi = pd.read_csv(CPI_FILE)
    yields = pd.read_csv(YIELD_FILE)

    print("=== AUDIT MACRO SIGNAL ===")

    print("\n1. Integritas jumlah baris")
    print("Signals:", len(signals))
    print("Yield:", len(yields))
    print("Tanggal yield duplikat:", yields["observation_date"].duplicated().sum())

    print("\n2. Kelengkapan waktu rilis CPI")
    cpi_nsa = cpi[cpi["series_id"] == "CPIAUCNS"].copy()
    verified = (
        cpi_nsa["release_time_verified"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    print("Total bulan CPI:", len(cpi_nsa))
    print("Waktu rilis terverifikasi:", int(verified.sum()))
    print("Waktu rilis belum terverifikasi:", int((~verified).sum()))

    print("\n3. Kelengkapan CPI pada sinyal")
    print("CPI tersedia:", signals["cpi_yoy_pct_nsa"].notna().sum())
    print("CPI kosong:", signals["cpi_yoy_pct_nsa"].isna().sum())

    print("\n4. Distribusi sinyal real yield")
    print(signals["real_yield_signal"].value_counts(dropna=False).sort_index())

    print("\n5. Lima baris awal dengan CPI tersedia")
    available = signals.dropna(subset=["cpi_yoy_pct_nsa"])
    print(available.head(5).to_string(index=False))

    print("\n6. Observasi terakhir")
    print(signals.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
