# XAUUSD Macro AI Agent

Proyek riset untuk analisis dan backtesting XAUUSD menggunakan struktur harga/support-resistance (SnR), tren multi-timeframe, dan eksperimen data makroekonomi. Prioritas proyek adalah validasi strategi dengan data historis, bukan eksekusi trading live.

## Status dan batasan

- Strategi masih dalam tahap riset; belum terbukti menghasilkan keuntungan secara konsisten.
- Hasil backtest bukan jaminan hasil masa depan dan tidak boleh langsung dianggap sebagai sinyal trading live.
- Data harga lokal **tidak otomatis diperbarui ketika terminal MT5 ditutup**. Dataset hanya diperbarui ketika skrip pengunduh dijalankan dan MT5 dapat diakses.
- MT5 tidak harus dibiarkan terbuka 24 jam hanya untuk keperluan backtest. Buka dan hubungkan MT5 saat ingin mengambil data terbaru, jalankan pengunduh, lalu terminal dapat ditutup setelah proses selesai.
- Skrip pengunduh saat ini menulis ulang file CSV timeframe yang berhasil diunduh. Selalu buat cadangan sebelum memperbarui data dan periksa rentang tanggal sebelum backtest.

## 1. Persiapan pertama kali

### Prasyarat

- Windows dan Python yang sesuai dengan proyek.
- MetaTrader 5 Exness terpasang.
- Akun/terminal MT5 berhasil login dan terhubung ke server broker.
- Simbol **XAUUSDm** tersedia.
- Repository sudah diunduh ke komputer.

Buka PowerShell, masuk ke folder proyek:

```powershell
cd C:\Users\SYAUQI\xau-macro-ai
```

Aktifkan virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Jika virtual environment belum tersedia, buat dan instal dependensi:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Jika PowerShell menolak aktivasi karena kebijakan skrip, gunakan terminal Command Prompt atau ikuti kebijakan eksekusi Windows yang berlaku di komputer. Jangan mengubah kebijakan keamanan secara permanen hanya untuk menjalankan proyek.

## 2. Prosedur rutin sebelum analisis atau backtest

Lakukan langkah-langkah ini setiap kali ingin memakai data terbaru.

### Langkah 1 — Sinkronkan kode proyek

Jika folder proyek sudah terhubung ke Git:

```powershell
git pull origin main
```

Jika ada perubahan lokal yang belum disimpan, periksa `git status` terlebih dahulu sebelum pull. Jangan menimpa perubahan lokal yang masih diperlukan.

### Langkah 2 — Buka dan hubungkan MT5

1. Buka MetaTrader 5 Exness.
2. Pastikan login ke akun dan server broker berhasil.
3. Pastikan simbol `XAUUSDm` tersedia di Market Watch.
4. Pastikan terminal dapat menerima data harga. Tunggu beberapa saat jika baru dibuka.

Terminal hanya perlu tersedia selama pengambilan data. Menyalakan MT5 sepanjang hari tidak akan memperbarui file CSV lokal secara otomatis tanpa menjalankan pengunduh.

### Langkah 3 — Buat cadangan dataset yang sekarang

Pengunduh saat ini menulis ulang file timeframe yang berhasil diunduh. Sebelum menjalankannya, buat cadangan:

```powershell
$backup = ".\data\backup\$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force $backup | Out-Null
Copy-Item .\data\raw\xauusdm_*.csv $backup -ErrorAction SilentlyContinue
Write-Host "Cadangan disimpan di $backup"
```

Jangan hapus cadangan sampai dataset baru selesai diperiksa.

### Langkah 4 — Unduh ulang data dari MT5

Jalankan:

```powershell
python .\src\download_intraday_mt5.py
```

Skrip mengunduh timeframe M5, M15, H1, dan H4 untuk simbol `XAUUSDm`, mulai dari 1 Januari 2020 sampai waktu saat skrip dijalankan, lalu menyimpan hasil ke:

- `data/raw/xauusdm_m5.csv`
- `data/raw/xauusdm_m15.csv`
- `data/raw/xauusdm_h1.csv`
- `data/raw/xauusdm_h4.csv`

Tunggu sampai skrip selesai. Untuk setiap timeframe, perhatikan jumlah candle, tanggal awal, tanggal akhir, jumlah duplikat, dan apakah muncul pesan `TIDAK ADA DATA`.

**Jangan lanjut ke analisis/backtest** jika unduhan gagal, rentang historis tiba-tiba jauh lebih pendek, jumlah candle tidak masuk akal, atau salah satu timeframe tidak berhasil diunduh. Bandingkan hasil dengan cadangan. Jika perlu, pulihkan file lama dari folder cadangan sebelum mencoba lagi.

### Langkah 5 — Pastikan file data benar-benar terbaru

Periksa waktu modifikasi dan ukuran file:

```powershell
Get-ChildItem .\data\raw\xauusdm_*.csv | Select-Object Name, Length, LastWriteTime
```

Periksa lima baris terakhir dan nama kolom setiap timeframe:

```powershell
@'
from pathlib import Path
import pandas as pd

for tf in ("m5", "m15", "h1", "h4"):
    path = Path("data/raw") / f"xauusdm_{tf}.csv"
    if not path.exists():
        print(f"\n{tf.upper()}: FILE TIDAK ADA")
        continue
    df = pd.read_csv(path)
    if df.empty or "time" not in df.columns:
        print(f"\n{tf.upper()}: DATA KOSONG ATAU KOLOM time TIDAK ADA")
        continue
    times = pd.to_datetime(df["time"], utc=True, errors="coerce")
    print(f"\n{tf.upper()}: rows={len(df):,}")
    print(f"Awal UTC: {times.min()}")
    print(f"Akhir UTC: {times.max()}")
    print(f"Duplikat waktu: {times.duplicated().sum()}")
    print(f"Kolom: {list(df.columns)}")
    print(df.tail(3).to_string(index=False))
'@ | python -
```

Untuk M5, waktu candle terakhir biasanya mendekati waktu pengunduhan, tetapi candle terakhir dapat berupa candle yang belum selesai atau waktu data terakhir yang tersedia dari broker. Jangan menuntut timestamp sama persis dengan jam komputer. Periksa juga apakah rentang historis masih mencakup periode lama yang dibutuhkan untuk backtest.

### Langkah 6 — Perbarui pemindaian SnR

Setelah keempat file diperiksa, jalankan:

```powershell
python .\src\snr_zone_scanner.py
```

Skrip menampilkan arah tren H4/H1, zona support/resistance terdekat, validitas data, dan pemeriksaan sinyal M5. Hasil disimpan di:

- `reports/snr_zones_latest.csv`
- `reports/snr_signals_latest.csv`

Perhatikan bagian **VALIDITAS DATA** dan **PEMERIKSAAN ENTRY M5**. Jika statusnya `STALE_DATA`, jangan gunakan sinyal itu; pastikan MT5 terhubung, ulangi unduhan, lalu jalankan scanner kembali. Zona yang berkonflik atau sinyal `NONE` bukan kesalahan yang harus dipaksa menjadi entry—hasilnya dapat berarti belum ada setup yang layak.

### Langkah 7 — Jalankan backtest SnR

Setelah data dan scanner lolos pemeriksaan:

```powershell
python .\src\backtest_snr_v1.py
```

Skrip menghasilkan laporan di folder `reports`, termasuk:

- `reports/snr_v1_summary.csv` — ringkasan performa.
- `reports/snr_v1_trades.csv` — daftar transaksi simulasi.

Baca ringkasan untuk periode **FULL** dan **OOS** (out-of-sample, mulai 1 Januari 2024). Periksa jumlah trade, win rate, expectancy dalam R, profit factor, net R, dan drawdown jika metrik tersebut tersedia. Jangan menilai strategi hanya dari win rate atau satu periode yang untung. Perhatikan juga apakah hasil berubah besar antara periode penuh dan OOS.

**Pemeriksaan penting spread:** backtest SnR V1 saat ini dapat mengisi spread dengan nol jika kolom `spread` tidak ada, dan mengubah nilai spread kosong menjadi nol. Kondisi itu dapat membuat simulasi terlalu optimistis. Sebelum mempercayai hasil, pastikan file M5 memiliki kolom `spread` dengan nilai valid dan tidak seluruhnya nol. Jika kolom tidak ada, nilainya kosong, atau seluruhnya nol, hentikan interpretasi hasil dan periksa sumber data/script terlebih dahulu.

## 3. Rutinitas singkat setiap kali ingin menguji ulang

Urutan rutin yang disarankan:

1. Buka MT5 Exness dan pastikan tersambung.
2. Jalankan `git pull origin main` untuk mengambil perubahan kode terbaru.
3. Buat cadangan CSV.
4. Jalankan `python .\src\download_intraday_mt5.py`.
5. Pastikan semua timeframe berhasil, rentang tanggal historis masih benar, dan candle terbaru masuk akal.
6. Jalankan `python .\src\snr_zone_scanner.py`; jangan gunakan hasil `STALE_DATA`.
7. Jalankan `python .\src\backtest_snr_v1.py`.
8. Simpan atau salin ringkasan hasil untuk membandingkan eksperimen berikutnya.

Langkah 4–7 perlu diulangi setelah data diperbarui. Membuka MT5 saja, tanpa menjalankan skrip pengunduh, **tidak** memperbarui CSV lokal.

## 4. Jika ingin data selalu terbarui

Untuk workflow backtesting saat ini, pembaruan manual sebelum analisis sudah cukup; MT5 tidak perlu menyala 24/7. Jika nanti diperlukan pembaruan terjadwal, buat tugas Windows Task Scheduler yang menjalankan pengunduh pada jadwal tertentu dan pastikan terminal MT5 sudah berjalan, login, dan bisa diakses saat tugas dimulai. Jadwal otomatis tidak boleh dianggap berhasil tanpa memeriksa log, timestamp data, serta cakupan historis setelah setiap pengunduhan.

Jangan menjadwalkan backtest secara otomatis sebelum validasi data dan asumsi spread selesai, agar laporan tidak diam-diam dibuat dari data tidak lengkap atau spread nol.

## 5. Struktur file penting

- `src/download_intraday_mt5.py` — unduh ulang candle MT5.
- `src/snr_zone_scanner.py` — pindai tren dan zona SnR terbaru.
- `src/backtest_snr_v1.py` — simulasi historis strategi SnR.
- `data/raw/` — dataset candle lokal; jangan unggah dataset privat ke repository.
- `data/backup/` — cadangan lokal sebelum pembaruan data.
- `reports/` — keluaran scanner dan laporan backtest.

## 6. Keamanan

Jangan commit atau mengunggah API key, kredensial broker, file `.env`, dataset privat, atau laporan yang mengandung informasi sensitif. Periksa `git status` sebelum commit.

## 7. Instalasi dependensi

Jika dependensi belum dipasang atau lingkungan Python baru dibuat:

```powershell
python -m pip install -r requirements.txt
```
