"""Test predictions of the latest run, without GeoCP: `run_pipeline.geocp` with geocp.enabled false writes only
snapshots/<run>/data/predictions_test.csv (the `--stage geocp` command would force the calibration on).

    python -m experiments.predict_test --override section.key=value ...   (the same overrides as train)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_parser import load_config  # noqa: E402
import run_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(a.override)
    cfg["geocp"]["enabled"] = False
    run_pipeline.geocp(cfg, None)


if __name__ == "__main__":
    main()
