"""GeoConformal Prediction.

The implementation lives in ``uncertainty/geocp.py``; this
shim keeps ``python geocp.py --alpha 0.10`` working from the project root."""
import runpy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    runpy.run_module("uncertainty.geocp", run_name="__main__")
else:
    from uncertainty.geocp import *      # noqa: F401,F403
