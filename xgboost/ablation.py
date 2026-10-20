"""Three-way ablation.

The implementation lives in ``training/ablation.py``; this
shim keeps ``python ablation.py --arms A B C`` working from the project root."""
import runpy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    runpy.run_module("training.ablation", run_name="__main__")
else:
    from training.ablation import *      # noqa: F401,F403
