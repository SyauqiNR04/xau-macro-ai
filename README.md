# XAUUSD Macro AI Agent

Riset strategi XAUUSD berbasis Support/Resistance (SnR), tren H4/H1, dan eksekusi M5. Prioritas: validasi historis dan pengendalian bias backtest; bukan trading live.

## Status

- Strategi belum terbukti profitable. Jangan gunakan hasil riset sebagai sinyal live.
- SnR aktif memakai pivot M15/H1 yang baru tersedia setelah candle kanan terkonfirmasi.
- Spread M5 wajib ada dan valid. Kolom hilang, nilai kosong/negatif, atau spread seluruhnya nol akan menghentikan backtest.
- Holdout final dikunci secara default dan tidak disimulasikan oleh runner biasa.
- Slippage diuji pada USD 0.05, 0.10, dan 0.20 per sisi. Bootstrap CI adalah interval percentile pada expectancy trade-level; ia tidak menjamin profitabilitas dan belum mengatasi semua dependensi serial.

## Setup Windows

```powershell
cd C:\Users\SYAUQI\xau-macro-ai
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-test.txt
```

Dependensi MetaTrader5 digunakan untuk mengambil data dari terminal Exness. Jangan commit kredensial, file .env, atau dataset privat.

## Alur riset yang benar

1. Sinkronkan kode dan backup data lokal.
2. Unduh data MT5 bila perlu: python .\src\download_intraday_mt5.py.
3. Jalankan audit data/spread: python .\scripts\audit\audit_data.py.
4. Jalankan tes lookahead: python -m pytest .\tests -q dan python .\scripts\audit\audit_lookahead.py.
5. Jalankan baseline SnR development: python .\src\backtest_snr_v1.py.
6. Periksa laporan development dan tahunannya sebelum mengubah parameter:
   - reports/snr_development_summary.csv
   - reports/snr_development_annual.csv
7. Baru setelah baseline ditinjau dan dibekukan, jalankan uji filter makro development:
   python .\scripts\audit\compare_macro_filter.py --confirm-baseline-reviewed.

Jangan memilih parameter berdasarkan holdout. Rentang development berakhir sebelum 2025-01-01; setiap trade yang melewati batas development dipurge dari laporan development.

## Holdout final — jangan dibuka saat pengembangan

Default config.yaml adalah holdout_enabled: false. Runner biasa tidak menjalankan simulator di holdout. Jangan ubah pengaturan ini sampai strategi, parameter, aturan biaya, dan kriteria keputusan dibekukan serta dicatat.

Hanya pada keputusan final:
1. Review dan simpan hasil development beserta konfigurasi/commit yang diuji.
2. Ubah validation.holdout_enabled menjadi true.
3. Jalankan python .\src\backtest_snr_v1.py --final-unseal tepat sekali.
4. Simpan reports/snr_holdout_final.csv dan reports/snr_holdout_annual_final.csv. Jangan kembali mengoptimasi parameter berdasarkan hasil holdout; jika diubah, holdout tersebut tidak lagi murni.

Holdout saat ini dikonfigurasi mulai 2025-01-01. Karena data historis pernah dipakai pada eksperimen strategi lain, sebut ini holdout khusus eksperimen SnR, bukan data yang secara universal belum pernah dilihat.

## Struktur utama

- config.yaml — parameter strategi, biaya, development, dan holdout.
- src/xau/data.py — loader dengan validasi spread gagal-keras dan alignment candle tertutup.
- src/xau/snr.py — pivot, zona SnR kausal, sinyal, dan simulasi barrier yang sebagian besar vektorisasi.
- src/xau/metrics.py — expectancy, bootstrap CI, laporan tahunan.
- scripts/run_snr_backtest.py — runner baseline dengan holdout terkunci.
- scripts/audit/audit_data.py — audit OHLC/spread.
- scripts/audit/audit_lookahead.py dan tests/ — tes lookahead serta biaya.
- scripts/audit/compare_macro_filter.py — eksperimen apakah filter makro meningkatkan expectancy, development saja, dengan konfirmasi eksplisit.
- src/backtest_snr_v1.py — entry point kompatibilitas untuk runner baru.
