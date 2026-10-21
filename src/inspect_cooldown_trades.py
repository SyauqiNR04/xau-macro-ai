import pandas as pd

path = "reports/cooldown_comparison_trades.csv"
df = pd.read_csv(path)

print("=== KOLOM DATA TRANSAKSI ===")
print(df.columns.tolist())

print("\n=== CONTOH 5 TRANSAKSI ===")
print(df.head().to_string(index=False))

print("\n=== JUMLAH TRANSAKSI PER MODEL DAN SKENARIO ===")
print(
    df.groupby(["model", "scenario"])
      .size()
      .rename("trades")
      .reset_index()
      .to_string(index=False)
)
