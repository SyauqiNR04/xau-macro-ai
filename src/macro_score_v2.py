
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SIGNAL_FILE = PROCESSED_DIR / "macro_signal.csv"
DOLLAR_FILE = PROCESSED_DIR / "dollar_features.csv"
OUTPUT_FILE = PROCESSED_DIR / "macro_score_v2.csv"

# Baseline eksperimen: bobot kedua faktor sama besar.
REAL_YIELD_WEIGHT = 0.50
DOLLAR_WEIGHT = 0.50

# Hindari penggunaan sinyal dolar yang sudah terlalu lama.
MAX_DOLLAR_AGE_DAYS = 4


def load_inputs():
    for path in [SIGNAL_FILE, DOLLAR_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    macro = pd.read_csv(SIGNAL_FILE, parse_dates=["observation_date"])
    dollar = pd.read_csv(DOLLAR_FILE, parse_dates=["observation_date"])

    required_macro = {"observation_date", "real_yield_signal"}
    required_dollar = {
        "observation_date",
        "dollar_index",
        "dollar_signal",
    }

    if not required_macro.issubset(macro.columns):
        raise ValueError(
            f"Kolom macro_signal.csv kurang: "
            f"{required_macro - set(macro.columns)}"
        )

    if not required_dollar.issubset(dollar.columns):
        raise ValueError(
            f"Kolom dollar_features.csv kurang: "
            f"{required_dollar - set(dollar.columns)}"
        )

    macro["real_yield_signal"] = pd.to_numeric(
        macro["real_yield_signal"], errors="coerce"
    )
    dollar["dollar_signal"] = pd.to_numeric(
        dollar["dollar_signal"], errors="coerce"
    )

    macro = macro.drop_duplicates(
        subset=["observation_date"], keep="last"
    ).sort_values("observation_date")

    dollar = dollar.drop_duplicates(
        subset=["observation_date"], keep="last"
    ).sort_values("observation_date")

    return macro, dollar


def build_score(macro, dollar):
    # Gunakan sinyal dolar dari tanggal observasi yang STRICTLY lebih awal.
    # Ini pendekatan konservatif untuk mengurangi risiko memakai data
    # dolar hari ini sebelum data tersebut benar-benar tersedia.
    dollar = dollar.rename(
        columns={"observation_date": "dollar_observation_date"}
    )

    merged = pd.merge_asof(
        macro,
        dollar[
            [
                "dollar_observation_date",
                "dollar_index",
                "dollar_signal",
            ]
        ].sort_values("dollar_observation_date"),
        left_on="observation_date",
        right_on="dollar_observation_date",
        direction="backward",
        allow_exact_matches=False,
    )

    merged["dollar_age_days"] = (
        merged["observation_date"] - merged["dollar_observation_date"]
    ).dt.days

    stale = merged["dollar_age_days"] > MAX_DOLLAR_AGE_DAYS
    merged.loc[stale, "dollar_signal"] = pd.NA

    # Validasi bahwa sinyal benar-benar hanya -1, 0, atau +1.
    valid_real_yield = merged["real_yield_signal"].isin([-1, 0, 1])
    valid_dollar = merged["dollar_signal"].isin([-1, 0, 1])

    merged["real_yield_score"] = (
        merged["real_yield_signal"] * 100
    ).where(valid_real_yield)

    merged["dollar_score"] = (
        merged["dollar_signal"] * 100
    ).where(valid_dollar)

    merged["available_factors"] = (
        valid_real_yield.astype(int) + valid_dollar.astype(int)
    )

    # Bobot dinormalisasi berdasarkan faktor yang tersedia.
    # Faktor yang hilang tidak dianggap sebagai sinyal netral.
    numerator = (
        merged["real_yield_score"].fillna(0) * REAL_YIELD_WEIGHT
        + merged["dollar_score"].fillna(0) * DOLLAR_WEIGHT
    )

    denominator = (
        valid_real_yield.astype(float) * REAL_YIELD_WEIGHT
        + valid_dollar.astype(float) * DOLLAR_WEIGHT
    )

    merged["macro_score_v2"] = (
        numerator / denominator.replace(0, float("nan"))
    )

    merged["signal_coverage"] = merged["available_factors"].map({
        0: "Data belum cukup",
        1: "Satu faktor tersedia",
        2: "Dua faktor tersedia",
    })

    merged["macro_bias_v2"] = "Netral"

    merged.loc[
        merged["macro_score_v2"] > 0,
        "macro_bias_v2",
    ] = "Cenderung bullish emas"

    merged.loc[
        merged["macro_score_v2"] < 0,
        "macro_bias_v2",
    ] = "Cenderung bearish emas"

    merged.loc[
        merged["available_factors"] == 0,
        "macro_bias_v2",
    ] = "Data belum cukup"

    merged["factor_agreement"] = "Tidak dapat dibandingkan"

    both = merged["available_factors"] == 2

    merged.loc[
        both
        & (
            merged["real_yield_signal"]
            == merged["dollar_signal"]
        ),
        "factor_agreement",
    ] = "Searah"

    merged.loc[
        both
        & (
            merged["real_yield_signal"]
            != merged["dollar_signal"]
        ),
        "factor_agreement",
    ] = "Berlawanan"

    columns = [
        "observation_date",
        "real_yield_signal",
        "real_yield_score",
        "dollar_observation_date",
        "dollar_age_days",
        "dollar_index",
        "dollar_signal",
        "dollar_score",
        "available_factors",
        "signal_coverage",
        "factor_agreement",
        "macro_score_v2",
        "macro_bias_v2",
    ]

    # Bawa konteks CPI jika tersedia dari Macro Signal sebelumnya.
    for col in [
        "cpi_reference_month",
        "cpi_yoy_pct_nsa",
        "cpi_yoy_change_pp",
    ]:
        if col in merged.columns:
            columns.append(col)

    return merged[columns].sort_values("observation_date")


def main():
    macro, dollar = load_inputs()
    result = build_score(macro, dollar)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n=== MACRO SCORE V2 SELESAI ===")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Jumlah baris: {len(result)}")
    print(
        "Tanggal duplikat:",
        result["observation_date"].duplicated().sum(),
    )

    print("\nCakupan faktor:")
    print(result["signal_coverage"].value_counts(dropna=False))

    print("\nKesepakatan faktor:")
    print(result["factor_agreement"].value_counts(dropna=False))

    print("\nDistribusi bias:")
    print(result["macro_bias_v2"].value_counts(dropna=False))

    print("\n10 observasi terbaru:")
    print(result.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
