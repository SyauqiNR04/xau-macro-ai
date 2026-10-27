"""Compatibility entry point; the maintained implementation lives in src/xau/."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts" / "run_snr_backtest.py"), run_name="__main__")
