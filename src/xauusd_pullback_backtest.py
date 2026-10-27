"""Backtest strategi pullback EMA XAUUSD pada timeframe M5.

Catatan:
- Satu unit backtesting mewakili 1 oz emas (setara 0,01 lot jika kontrak 1 lot = 100 oz).
- CSV timestamp tanpa zona waktu diasumsikan UTC; timestamp berzona dikonversi ke UTC.
- Spread yang diterima Backtest adalah proporsi harga, bukan USD langsung.

Contoh:
    python .\\src\\xauusd_pullback_backtest.py --csv .\\data\\raw\\xauusdm_m5.csv
    python .\\src\\xauusd_pullback_backtest.py --csv .\\data\\raw\\xauusdm_m5.csv --spread 0.00006
    python .\\src\\xauusd_pullback_backtest.py --csv .\\data\\raw\\xauusdm_m5.csv --optimize
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy


def ema(series, n):
    return pd.Series(series).ewm(span=n, adjust=False).mean().to_numpy()


def atr(high, low, close, n=14):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_c = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


class EmaPullbackScalp(Strategy):
    ema_fast = 20
    ema_mid = 50
    ema_slow = 200
    atr_period = 14
    sl_atr = 1.5
    rr = 1.5
    risk_pct = 0.005
    max_atr = 6.0  # USD per ounce; tune only after validating out-of-sample
    session_start = 7  # UTC
    session_end = 16  # UTC
    max_leverage = 30

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.e_fast = self.I(ema, c, self.ema_fast)
        self.e_mid = self.I(ema, c, self.ema_mid)
        self.e_slow = self.I(ema, c, self.ema_slow)
        self.atr = self.I(atr, h, l, c, self.atr_period)

    def _size_oz(self, price, sl_dist):
        """Position size in ounces; return 0 rather than force excess risk."""
        if not np.isfinite(sl_dist) or sl_dist <= 0 or price <= 0:
            return 0
        risk_amount = self.equity * self.risk_pct
        risk_limited_oz = int(risk_amount / sl_dist)
        leverage_limited_oz = int(self.equity * self.max_leverage / price)
        return max(0, min(risk_limited_oz, leverage_limited_oz))

    def next(self):
        if self.position:
            return

        # load_csv normalizes all timestamps to UTC before this session filter.
        hour = self.data.index[-1].hour
        if not (self.session_start <= hour < self.session_end):
            return

        price = self.data.Close[-1]
        open_ = self.data.Open[-1]
        low = self.data.Low[-1]
        high = self.data.High[-1]
        a = self.atr[-1]
        if not np.isfinite(a) or a <= 0 or a > self.max_atr:
            return

        sl_dist = self.sl_atr * a
        size = self._size_oz(price, sl_dist)
        if size < 1:
            return

        uptrend = self.e_fast[-1] > self.e_mid[-1] > self.e_slow[-1]
        downtrend = self.e_fast[-1] < self.e_mid[-1] < self.e_slow[-1]

        # Sinyal diketahui setelah candle tutup; market order dieksekusi bar berikutnya.
        if uptrend and low <= self.e_fast[-1] and price > self.e_fast[-1] and price > open_:
            self.buy(size=size, sl=price - sl_dist, tp=price + sl_dist * self.rr)
        elif downtrend and high >= self.e_fast[-1] and price < self.e_fast[-1] and price < open_:
            self.sell(size=size, sl=price + sl_dist, tp=price - sl_dist * self.rr)


def load_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File CSV tidak ditemukan: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Prioritaskan kombinasi date + time bila CSV MT5 memisahkan keduanya.
    if "date" in df.columns and "time" in df.columns:
        timestamps = pd.to_datetime(
            df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
            errors="coerce", utc=True,
        )
    else:
        time_col = next(
            (c for c in ("datetime", "timestamp", "time", "date") if c in df.columns),
            None,
        )
        if time_col is None:
            raise ValueError("Kolom waktu tidak ditemukan (time/datetime/date/timestamp).")
        timestamps = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    required = ("open", "high", "low", "close")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom OHLC wajib tidak ditemukan: {', '.join(missing)}")

    volume_col = "volume" if "volume" in df.columns else "tick_volume" if "tick_volume" in df.columns else None
    clean = pd.DataFrame(index=pd.DatetimeIndex(timestamps, name="datetime"))
    for col in required:
        clean[col.capitalize()] = pd.to_numeric(df[col], errors="coerce").to_numpy()
    clean["Volume"] = (
        pd.to_numeric(df[volume_col], errors="coerce").to_numpy()
        if volume_col else 0.0
    )

    clean = clean.loc[~clean.index.isna()].sort_index()
    clean = clean[~clean.index.duplicated(keep="last")].dropna(subset=["Open", "High", "Low", "Close"])
    clean = clean[(clean[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    clean = clean[clean["High"] >= clean["Low"]]
    if clean.empty:
        raise ValueError("Tidak ada candle OHLC valid setelah pembersihan CSV.")
    return clean[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def make_backtest(df, cash, spread, commission, leverage):
    return Backtest(
        df,
        EmaPullbackScalp,
        cash=cash,
        spread=spread,
        commission=commission,
        margin=1 / leverage,
        exclusive_orders=True,
        finalize_trades=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Backtest pullback EMA XAUUSD M5")
    parser.add_argument("--csv", required=True, help="Path CSV candle XAUUSD M5")
    parser.add_argument("--cash", type=float, default=100_000, help="Modal simulasi dalam mata uang akun")
    parser.add_argument(
        "--spread", type=float, default=0.00006,
        help="Spread proporsional; 0.00006 kira-kira 0.24 USD pada harga emas 4000 USD",
    )
    parser.add_argument("--commission", type=float, default=0.0,
                        help="Komisi proporsional per transaksi menurut format backtesting.py")
    parser.add_argument("--leverage", type=float, default=30.0,
                        help="Leverage simulasi; margin akan disamakan dengan 1/leverage")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.cash <= 0 or args.leverage <= 0:
        parser.error("--cash dan --leverage harus lebih besar dari nol.")
    if args.spread < 0 or args.commission < 0:
        parser.error("--spread dan --commission tidak boleh negatif.")

    df = load_csv(args.csv)
    if len(df) < 250:
        raise ValueError(f"Data hanya {len(df)} candle; terlalu sedikit untuk EMA200 dan evaluasi yang layak.")
    print(f"Data: {len(df):,} candle | {df.index[0]} s/d {df.index[-1]} (UTC)")
    print(f"Spread asumsi: {args.spread:.6f} proporsional | leverage: 1:{args.leverage:g}")
    print("Catatan: spread statis bukan spread historis MT5; hasil belum mencakup slippage.")

    bt = make_backtest(df, args.cash, args.spread, args.commission, args.leverage)

    if args.optimize:
        split = int(len(df) * 0.7)
        if split < 250 or len(df) - split < 100:
            raise ValueError("Data terlalu sedikit untuk pembagian 70% in-sample dan 30% out-of-sample.")
        bt_in = make_backtest(df.iloc[:split], args.cash, args.spread, args.commission, args.leverage)
        best = bt_in.optimize(
            sl_atr=[1.0, 1.5, 2.0],
            rr=[1.0, 1.5, 2.0],
            ema_fast=[10, 20, 30],
            maximize="SQN",
            constraint=lambda p: p.ema_fast < p.ema_mid < p.ema_slow,
        )
        print("\n=== HASIL IN-SAMPLE (70% awal) ===")
        print(best)
        params = best._strategy
        print(f"Parameter terpilih: EMA fast={params.ema_fast}, SL ATR={params.sl_atr}, RR={params.rr}")

        bt_out = make_backtest(df.iloc[split:], args.cash, args.spread, args.commission, args.leverage)
        stats_out = bt_out.run(sl_atr=params.sl_atr, rr=params.rr, ema_fast=params.ema_fast,
                               max_leverage=args.leverage)
        print("\n=== HASIL OUT-OF-SAMPLE (30% akhir; parameter tidak dioptimasi ulang) ===")
        print(stats_out)
        return

    stats = bt.run(max_leverage=args.leverage)
    print("\n=== HASIL BACKTEST ===")
    print(stats)

    trades = stats["_trades"]
    if len(trades):
        wins = trades.loc[trades["PnL"] > 0, "PnL"]
        losses = trades.loc[trades["PnL"] <= 0, "PnL"]
        winrate = len(wins) / len(trades)
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = losses.mean() if len(losses) else 0.0
        expectancy = trades["PnL"].mean()
        print("\n--- Ringkasan tambahan ---")
        print(f"Jumlah trade       : {len(trades)}")
        print(f"Win rate           : {winrate:.2%}")
        print(f"Rata-rata profit   : {avg_win:,.2f}")
        print(f"Rata-rata loss     : {avg_loss:,.2f}")
        print(f"Expectancy/trade   : {expectancy:,.2f}")
    else:
        print("Tidak ada trade. Periksa rentang data, timestamp UTC, dan filter strategi.")

    if args.plot:
        bt.plot()


if __name__ == "__main__":
    main()
