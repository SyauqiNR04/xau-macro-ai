"""Incrementally refresh the live SnR state without replaying the historical zone loop.

The indicator and pivot candidates are recomputed from the CSV history, but the
causal zone/signal loop starts at the last stored closed M5 candle. On first
run, only a recent bootstrap window is evaluated. SQLite stores processed
signals and the latest zone snapshot.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xau.snr import build_signals


SIGNAL_COLUMNS = [
    "time", "signal_time", "close", "signal", "zone_score", "zone_kind",
    "zone_low", "zone_high", "zone_mid", "zone_pivots", "zone_timeframes",
    "h4_bias", "h1_bias", "current_zones_json",
]


def _utc_iso(value) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat()


def _latest_closed_signal_time(m5_path: Path) -> pd.Timestamp:
    times = pd.to_datetime(pd.read_csv(m5_path, usecols=["time"])["time"], utc=True, errors="coerce").dropna()
    if times.empty:
        raise ValueError(f"No valid candle timestamps found in {m5_path}")
    now = pd.Timestamp.now(tz="UTC")
    closed_opens = times.loc[times + pd.Timedelta(minutes=5) <= now]
    if closed_opens.empty:
        raise ValueError("No closed M5 candle is available yet.")
    return closed_opens.max() + pd.Timedelta(minutes=5)


def _meta_get(conn, key):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(conn, key, value):
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def _bias(bull, bear) -> str:
    if bool(bull):
        return "BULLISH"
    if bool(bear):
        return "BEARISH"
    return "NEUTRAL"


def _db_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return _utc_iso(value)
    if hasattr(value, "item"):
        value = value.item()
    return value


def _print_latest(conn, prefix=""):
    row = conn.execute(
        "SELECT signal_time, close, signal, zone_score, zone_kind, zone_low, zone_high, "
        "h4_bias, h1_bias, current_zones_json FROM signals ORDER BY signal_time DESC LIMIT 1"
    ).fetchone()
    if not row:
        print(prefix + "Cache belum memiliki sinyal tersimpan.")
        return
    signal_name = {1: "LONG", -1: "SHORT", 0: "NO SIGNAL"}.get(row[2], "NO SIGNAL")
    print(prefix + f"Latest closed M5: {row[0]}")
    print(f"Close: {row[1]} | H4: {row[7]} | H1: {row[8]} | Signal: {signal_name}")
    if row[4]:
        print(f"Signal zone: {row[4]} {row[5]}–{row[6]} | score={row[3]}")
    print(f"Current nearby zones: {row[9] or '[]'}")


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally update closed-M5 SnR state into SQLite."
    )
    parser.add_argument("--bootstrap-days", type=int, default=7,
                        help="How many recent days of signals to seed on first run (default: 7).")
    parser.add_argument("--rebuild", action="store_true",
                        help="Clear cached signals and seed the recent bootstrap window again.")
    args = parser.parse_args()
    if args.bootstrap_days < 1:
        parser.error("--bootstrap-days must be at least 1")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data_cfg, strat, execution = cfg["data"], cfg["strategy"], cfg["execution"]
    raw_dir = ROOT / data_cfg["raw_dir"]
    m5_path = raw_dir / f"{data_cfg['symbol_file']}_m5.csv"
    if not m5_path.exists():
        raise FileNotFoundError(f"Missing M5 data: {m5_path}")

    db_path = ROOT / data_cfg["processed_dir"] / "snr_state.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS signals (
            time TEXT PRIMARY KEY,
            signal_time TEXT NOT NULL,
            close REAL,
            signal INTEGER NOT NULL,
            zone_score INTEGER NOT NULL,
            zone_kind TEXT NOT NULL,
            zone_low REAL,
            zone_high REAL,
            zone_mid REAL,
            zone_pivots INTEGER,
            zone_timeframes INTEGER,
            h4_bias TEXT NOT NULL,
            h1_bias TEXT NOT NULL,
            current_zones_json TEXT NOT NULL
        )"""
    )
    conn.commit()

    signature = {
        "symbol_file": data_cfg["symbol_file"],
        "pivot_left": strat["pivot_left"],
        "pivot_right": strat["pivot_right"],
        "max_pivot_age_days": strat["max_pivot_age_days"],
        "zone_tolerance_atr": strat["zone_tolerance_atr"],
        "min_zone_pivots": strat["min_zone_pivots"],
        "max_zone_width_atr": strat["max_zone_width_atr"],
        "min_body_range_ratio": strat["min_body_range_ratio"],
        "spread_points_multiplier": execution["spread_points_multiplier"],
    }
    signature_json = json.dumps(signature, sort_keys=True)
    previous_signature = _meta_get(conn, "strategy_signature")
    if args.rebuild or (previous_signature and previous_signature != signature_json):
        conn.execute("DELETE FROM signals")
        conn.execute("DELETE FROM meta WHERE key = 'last_processed_signal_time'")
        if previous_signature and previous_signature != signature_json:
            print("Konfigurasi strategi berubah; cache sinyal direset.")
        elif args.rebuild:
            print("Cache sinyal direset sesuai --rebuild.")
    _meta_set(conn, "strategy_signature", signature_json)
    conn.commit()

    latest_closed = _latest_closed_signal_time(m5_path)
    last_processed_text = _meta_get(conn, "last_processed_signal_time")
    last_processed = pd.to_datetime(last_processed_text, utc=True) if last_processed_text else None

    if last_processed is not None and latest_closed <= last_processed:
        print("Tidak ada candle M5 tertutup yang baru; perhitungan dilewati.")
        _print_latest(conn)
        conn.close()
        return

    if last_processed is None:
        start_time = latest_closed - pd.Timedelta(days=args.bootstrap_days)
        print(f"Bootstrap: memproses zona/sinyal untuk {args.bootstrap_days} hari terakhir.")
    else:
        start_time = last_processed
        print(f"Incremental update dari checkpoint {last_processed}.")

    started = time.perf_counter()
    frame = build_signals(
        raw_dir,
        data_cfg["symbol_file"],
        left=strat["pivot_left"],
        right=strat["pivot_right"],
        max_age_days=strat["max_pivot_age_days"],
        tolerance_atr=strat["zone_tolerance_atr"],
        min_pivots=strat["min_zone_pivots"],
        max_width_atr=strat["max_zone_width_atr"],
        min_body_range_ratio=strat["min_body_range_ratio"],
        spread_points_multiplier=execution["spread_points_multiplier"],
        start_time=start_time,
        end_time=latest_closed,
    )
    new = frame.loc[
        (frame["signal_time"] > start_time) &
        (frame["signal_time"] <= latest_closed)
    ].copy()
    if new.empty:
        print("Tidak ditemukan candle baru yang bisa diproses; checkpoint tidak diubah.")
        _print_latest(conn)
        conn.close()
        return

    records = []
    for row in new.itertuples(index=False):
        records.append((
            _utc_iso(row.time),
            _utc_iso(row.signal_time),
            float(row.close),
            int(row.signal),
            int(row.zone_score),
            str(row.zone_kind or ""),
            _db_value(row.zone_low),
            _db_value(row.zone_high),
            _db_value(row.zone_mid),
            int(row.zone_pivots),
            int(row.zone_timeframes),
            _bias(row.h4_bull, row.h4_bear),
            _bias(row.h1_bull, row.h1_bear),
            str(row.current_zones_json or "[]"),
        ))

    placeholders = ",".join("?" for _ in SIGNAL_COLUMNS)
    conn.executemany(
        f"INSERT INTO signals({','.join(SIGNAL_COLUMNS)}) VALUES({placeholders}) "
        "ON CONFLICT(time) DO UPDATE SET " +
        ",".join(f"{column}=excluded.{column}" for column in SIGNAL_COLUMNS if column != "time"),
        records,
    )
    _meta_set(conn, "last_processed_signal_time", _utc_iso(new["signal_time"].max()))
    conn.commit()

    print(f"Candle baru diproses: {len(new):,}")
    print(f"Durasi update: {time.perf_counter() - started:.1f} detik")
    print(f"SQLite state: {db_path}")
    _print_latest(conn)
    conn.close()


if __name__ == "__main__":
    main()
