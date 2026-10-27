# Incremental SnR state updater

## Purpose

`scripts/update_snr_state.py` maintains a local SQLite cache for live/monitoring
use. It evaluates the causal zone/signal loop only for candles after the last
stored closed-M5 checkpoint. On first run it seeds only a recent signal window
(default seven days), while reconstructing active pivots from the available
history. The database is written under `data/processed/snr_state.sqlite3`,
which is local runtime state and should not be committed.

## Run

From the repository root with the virtual environment active:

```powershell
python .\scripts\update_snr_state.py
```

Run the same command again after new closed M5 candles have arrived. If there
is no new closed M5 candle, the script exits without rebuilding indicators,
pivots, or zones.

Optional first-run history window:

```powershell
python .\scripts\update_snr_state.py --bootstrap-days 14
```

Reset the cached signal history and seed it again:

```powershell
python .\scripts\update_snr_state.py --rebuild
```

## What is and is not incremental

- The expensive candle-by-candle zone/signal loop starts after the saved
  checkpoint. The active pivot set is reconstructed at that checkpoint from
  confirmed pivots and their expiry times, so valid pivots within the configured
  maximum age remain available.
- The updater persists signal rows and the latest nearby-zone snapshot in
  SQLite. It only processes M5 candles whose close time has passed.
- EMA/ATR calculations and pivot candidate discovery are still recalculated
  from the CSV history when new candles arrive. They are vectorized, but this
  is not yet a fully incremental indicator engine.
- If old CSV history is corrected, use `--rebuild`. Normal incremental mode
  assumes existing history is append-only.
- This updater does not place trades or connect to MT5. It is a local signal
  state/cache step; execution remains separate.

## Validation

`tests/test_incremental_snr.py` checks that the signals and selected zones
produced by the incremental tail match the same time slice from a full build on
a deterministic synthetic dataset. Run:

```powershell
python -m unittest discover -s tests -v
```
