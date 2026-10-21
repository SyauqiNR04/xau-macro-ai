import MetaTrader5 as mt5
import pandas as pd

TERMINAL = r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
CANDIDATES = ["XAUUSDm", "XAUUSD247m"]

if not mt5.initialize(TERMINAL, timeout=60000):
    raise RuntimeError(f"MT5 gagal: {mt5.last_error()}")

try:
    for name in CANDIDATES:
        print(f"\n{'=' * 45}\nSYMBOL: {name}")

        info = mt5.symbol_info(name)
        if info is None:
            print("Simbol tidak tersedia.")
            continue

        print("Deskripsi:", info.description)
        print("Base/profit currency:", info.currency_base, "/", info.currency_profit)
        print("Digits:", info.digits, "| Point:", info.point)
        print("Contract size:", info.trade_contract_size)

        if not mt5.symbol_select(name, True):
            print("Gagal memilih simbol:", mt5.last_error())
            continue

        tick = mt5.symbol_info_tick(name)
        if tick:
            print("Bid:", tick.bid, "| Ask:", tick.ask)

        rates = mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_M5, 0, 1000)
        if rates is None or len(rates) == 0:
            print("Data M5 tidak tersedia:", mt5.last_error())
            continue

        df = pd.DataFrame(rates)
        df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)

        print("Jumlah candle:", len(df))
        print("Candle tertua UTC:", df["time_utc"].min())
        print("Candle terbaru UTC:", df["time_utc"].max())
        print("Close terakhir:", df.iloc[-1]["close"])
        print("Spread candle terakhir (points):", df.iloc[-1]["spread"])

finally:
    mt5.shutdown()
