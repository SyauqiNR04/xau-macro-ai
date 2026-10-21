
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

SCORE_FILE = DATA_DIR / "macro_score.csv"
SIGNAL_FILE = DATA_DIR / "macro_signal.csv"


def main():
    for path in (SCORE_FILE, SIGNAL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    scores = pd.read_csv(SCORE_FILE)
    signals = pd.read_csv(SIGNAL_FILE)

    print("=== AUDIT MACRO SCORE ===")

    # 1. Integritas baris dan tanggal
    print("\n1. Integritas data")
    print("Jumlah baris skor:", len(scores))
    print("Jumlah baris sinyal:", len(signals))
    print(
        "Tanggal observasi duplikat:",
        scores["observation_date"].duplicated().sum(),
    )

    assert len(scores) == len(signals), (
        "Jumlah baris skor berbeda dari sinyal."
    )

    # 2. Konsistensi arah skor
    print("\n2. Konsistensi arah skor")

    signal = pd.to_numeric(
        scores["real_yield_signal"], errors="coerce"
    )
    score = pd.to_numeric(
        scores["real_yield_score"], errors="coerce"
    )

    expected_score = signal * 100
    comparable = score.notna() & expected_score.notna()
    mismatches = (
        score[comparable] != expected_score[comparable]
    )

    print("Skor yang diperiksa:", int(comparable.sum()))
    print("Skor tidak konsisten:", int(mismatches.sum()))

    assert mismatches.sum() == 0, (
        "Arah skor tidak konsisten dengan real_yield_signal."
    )

    # 3. Validitas timestamp CPI
    print("\n3. Validitas timestamp CPI")

    available = pd.to_datetime(
        scores["available_at_utc"], utc=True, errors="coerce"
    )
    yield_time = pd.to_datetime(
        scores["yield_available_at_utc"],
        utc=True,
        errors="coerce",
    )

    has_cpi = available.notna()
    future_cpi = has_cpi & (available > yield_time)

    print("Baris dengan CPI tersedia:", int(has_cpi.sum()))
    print("CPI bertimestamp masa depan:", int(future_cpi.sum()))
    print(
        "Timestamp CPI tidak valid:",
        int((scores["cpi_timestamp_valid"] == False).sum()),
    )

    assert future_cpi.sum() == 0, (
        "Ditemukan CPI dengan timestamp setelah waktu yield."
    )

    # 4. Nilai kosong
    print("\n4. Kelengkapan data")
    for column in [
        "real_yield_score",
        "macro_score",
        "cpi_yoy_pct_nsa",
        "cpi_yoy_change_pp",
        "cpi_reference_month",
    ]:
        print(f"{column}: {int(scores[column].isna().sum())} kosong")

    # 5. Distribusi skor
    print("\n5. Distribusi macro bias")
    print(scores["macro_bias"].value_counts(dropna=False).to_string())

    # 6. Ringkasan timestamp
    print("\n6. Timestamp yield")
    print("Timestamp yield kosong:", int(yield_time.isna().sum()))
    print(
        "Basis timestamp:",
        scores["yield_timestamp_basis"].value_counts(
            dropna=False
        ).to_dict(),
    )

    print("\n=== AUDIT SELESAI ===")
    print(
        "Catatan: lolos audit teknis tidak otomatis membuktikan "
        "model bebas look-ahead bias."
    )


if __name__ == "__main__":
    main()
