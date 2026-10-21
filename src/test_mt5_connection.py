import MetaTrader5 as mt5

TERMINAL = r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"

try:
    print("Menghubungkan Python ke MT5 Exness...")

    if not mt5.initialize(TERMINAL, timeout=60000):
        print("KONEKSI GAGAL")
        print("Error:", mt5.last_error())
        raise SystemExit(1)

    print("KONEKSI BERHASIL")
    print("Versi MT5:", mt5.version())

    terminal = mt5.terminal_info()
    account = mt5.account_info()

    print("Terminal terhubung:", terminal.connected if terminal else None)
    print("Akun terdeteksi:", account is not None)
    print("Trade server:", account.server if account else "Tidak tersedia")

    symbols = mt5.symbols_get(group="*XAU*")

    print("\nSimbol terkait XAU:")
    if symbols:
        for symbol in symbols[:20]:
            print("-", symbol.name)
    else:
        print("Belum ditemukan; periksa simbol di Market Watch.")

    if symbols:
        symbol_name = next(
            (s.name for s in symbols if s.name == "XAUUSD"),
            symbols[0].name
        )

        if not mt5.symbol_select(symbol_name, True):
            print("Gagal memilih simbol:", symbol_name, mt5.last_error())
        else:
            rates = mt5.copy_rates_from_pos(
                symbol_name, mt5.TIMEFRAME_M5, 0, 5
            )

            print(f"\n5 candle M5 terakhir: {symbol_name}")
            if rates is not None and len(rates) > 0:
                for candle in rates:
                    print(
                        mt5.time.strftime("%Y-%m-%d %H:%M:%S", mt5.time.gmtime(candle["time"]))
                        if False else candle
                    )
            else:
                print("Data candle belum tersedia:", mt5.last_error())

finally:
    mt5.shutdown()
