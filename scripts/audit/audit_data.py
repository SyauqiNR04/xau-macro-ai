from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from xau.data import load_bars


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    d, ex = cfg["data"], cfg["execution"]
    for tf in ("m5", "m15", "h1", "h4"):
        path = ROOT / d["raw_dir"] / f"{d['symbol_file']}_{tf}.csv"
        df = load_bars(path, require_spread=(tf == "m5"),
                       spread_points_multiplier=ex["spread_points_multiplier"])
        print(f"{tf.upper()}: rows={len(df):,}; first={df.time.min()}; last={df.time.max()}")
        if tf == "m5":
            print(f"spread median price={df.spread_price.median():.4f}; zero rows={(df.spread_price == 0).sum():,}")


if __name__ == "__main__":
    main()
