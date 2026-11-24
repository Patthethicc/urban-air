"""Builds urban-air/amouei_ablation.ipynb (the Kaggle notebook of amouei-model/experiments)."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# Amouei (2023) ablation: which parts of the framework improve the model?

Code: `amouei-model/` (a copy of `model/` with the Amouei inputs added as switches, all **off** by default, so `baseline` is the main model).
Each experiment changes **one** thing against `baseline`, trains on the same stations and is scored on the **same 16 pinned test stations**:

| experiment | Amouei element tested |
|---|---|
| `year` | time of year (sin/cos day of year) |
| `weather` | temperature, humidity, wind vector at the lag hours (ERA5 via Open-Meteo, nearest point) |
| `weather_lead1` | the same, but the last step carries the **target hour's** weather |
| `road_density` | traffic proxy: street share of the 320 m patch |
| `lcz_proxy` | an LCZ-like class per station (one-hot), the stand-in for Amouei's LCZ stage |
| `morphology_tabular` | morphology band means fed to the LSTM as columns (Amouei style) instead of a CNN |
| `window24` | 24-hour input window instead of 3 hours |
| `amouei_lstm` | Amouei's network: LSTM 256-128-64-32, dense 128, dropout 0.5 |
| `amouei_lstm_w24` | Amouei's network with the 24-hour window |
| `amouei_full` | all of the above that Amouei used, together |
| `window12`, `window48` | how the gain depends on the window length |
| `recommended` | the elements that helped in the local screen, together: 24 h window + weather incl. the target hour + time of year |

The comparison is made on the test rows **every** experiment kept (a 24-hour window drops more samples), and the R² gain over `baseline` gets a 95 % interval from a bootstrap
over test stations. **Read a gain as real only when the whole interval is above 0**, and when it repeats with a second seed (`SEEDS = [42, 43]`).

**Kaggle setup:** GPU on, Internet on (to clone the code and download the weather). Attach your data dataset (the one with `Morphology/`, `Combined Dataset/`, optionally `Weather/`).
The notebook resumes: finished experiments are skipped. If time runs out, *Save Version*, add the output as an Input and run again.
""")

md("## 1. Settings (the only cell you edit)")
code(r'''
from pathlib import Path

EXPERIMENTS = ["baseline", "window24", "weather_lead1", "recommended", "weather", "year", "amouei_lstm", "road_density",
               "lcz_proxy", "morphology_tabular", "window12", "amouei_lstm_w24", "amouei_full"]   # most informative first, in case time runs out
               # optional extras: "window48", "cnn_baseline", "cnn_weather_lead1" (the CNN-LSTM ones are much slower)
SEEDS = [42]                  # [42, 43] repeats everything with a second seed: a gain that does not repeat is noise
QUICK = False                 # True = the local screening settings (no CV, every 4th row, 6 epochs): a fast look, not a result
ARM = "from_2023_with_preinstall"
RASTER = "master_raster_v3.tif"
TIME_BUDGET_H = 11.0          # stop starting new experiments after this (Kaggle ends a session at 12 h)

DATA = Path("/kaggle/input/datasets/patrickhansperez/urban-air-data-recent")   # your data dataset (path from the Input panel)
GITHUB = "https://github.com/Patthethicc/urban-air.git"                        # must contain amouei-model/ (commit and push it first)
FETCH_WEATHER = True          # download data/Weather/ from Open-Meteo if your dataset does not have it (needs Internet on)
''')

md("## 2. Code, data and libraries")
code(r'''
import json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd

T0 = time.time()
ON_KAGGLE = Path("/kaggle/working").exists()
if ON_KAGGLE:
    REPO = Path("/kaggle/working/urban-air")
else:                                            # local: open the notebook from the repository root
    REPO = Path.cwd() if (Path.cwd() / "amouei-model").is_dir() else Path.cwd().parent
AM = REPO / "amouei-model"

if ON_KAGGLE and not (AM / "run_pipeline.py").is_file():
    saved = next((p.parent.parent for p in Path("/kaggle/input").glob("**/amouei-model/run_pipeline.py")), None)
    if saved is not None:                         # an earlier session's output: code AND finished runs
        shutil.copytree(saved, REPO, dirs_exist_ok=True)
        print("restored from", saved)
    else:
        subprocess.run(["git", "clone", "--depth", "1", GITHUB, str(REPO)], check=True)
        print("cloned", GITHUB)
if not (AM / "run_pipeline.py").is_file():
    raise SystemExit(f"{AM} has no run_pipeline.py: push amouei-model/ to {GITHUB}, or attach a saved output that has it")

if ON_KAGGLE and not (REPO / "data" / "Morphology" / RASTER).is_file():
    found = list(DATA.rglob(RASTER)) if DATA.exists() else []
    if not found:
        raise SystemExit(f"{RASTER} not found under {DATA}: check DATA")
    src = found[0].parent.parent
    for part in (Path("Morphology"), Path("Boundaries"), Path("Combined Dataset") / ARM, Path("Weather")):
        if (src / part).is_dir():
            shutil.copytree(src / part, REPO / "data" / part, dirs_exist_ok=True)
    print("data copied from", src)

weather = REPO / "data" / "Weather" / "open_meteo_ncr_hourly.csv"
if not weather.is_file() and FETCH_WEATHER:
    subprocess.run([sys.executable, "-m", "preprocessing.fetch_weather", "--end", "2026-11-18"], cwd=AM, check=True)
print("weather file:", weather, "ok" if weather.is_file() else "MISSING (weather experiments will fail)")

import importlib.util
need = {"rasterio": "rasterio", "geopandas": "geopandas", "pyproj": "pyproj", "shapely": "shapely", "affine": "affine",
        "yaml": "pyyaml", "sklearn": "scikit-learn", "skopt": "scikit-optimize", "libpysal": "libpysal", "esda": "esda"}
missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
for p in (REPO / "data" / "Combined Dataset" / ARM / "locations.csv", AM / "configs" / "pinned_partition_2026-11.json"):
    assert p.is_file(), f"missing {p}"
print("ready")
''')

md("""
## 3. Check the code (about 4 minutes)
The 516 tests of `amouei-model/` (the main model's 500, and 16 for the Amouei switches). A failure here stops the notebook before hours of training.
""")
code(r'''
RUN_TESTS = True
if RUN_TESTS:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=AM, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise SystemExit("tests failed")
''')

md("""
## 4. Run the experiments
One at a time, each skipped if it finished in an earlier session. Logs: `amouei-model/experiments/logs/<experiment>_s<seed>.log`.
""")
code(r'''
flags = ["--seeds", *map(str, SEEDS), "--arm", ARM, "--raster", RASTER] + (["--quick"] if QUICK else [])
for exp in EXPERIMENTS:
    left = TIME_BUDGET_H - (time.time() - T0) / 3600
    if left < 0.25:
        print(f"time budget used up before {exp}: Save Version, attach the output as an Input, run again to continue")
        break
    print(f"=== {exp} ({left:.1f} h left)", flush=True)
    r = subprocess.run([sys.executable, "-u", "-m", "experiments.run_ablation", "--exp", exp, *flags], cwd=AM)
    if r.returncode != 0:
        print(f"{exp} returned {r.returncode}; the next experiment still runs")
''')

md("## 5. The comparison")
code(r'''
subprocess.run([sys.executable, "-m", "experiments.run_ablation", "--compare-only", *flags, "--bootstrap", "2000"], cwd=AM, check=True)
stem = "ablation_quick" if QUICK else "ablation"
table = pd.read_csv(AM / "experiments" / "results" / f"{stem}.csv")
info = json.loads((AM / "experiments" / "results" / f"{stem}_info.json").read_text())
print(f"{info['common_test_rows']:,} common test rows on {info['test_stations']} stations; previous-hour persistence R2 {info['persistence_r2']:.3f}")
pd.set_option("display.width", 220, "display.max_colwidth", 60)
display(table[["experiment", "seed", "test_r2", "r2_gain", "gain_ci_low", "gain_ci_high", "test_rmse", "rmse_above_30", "parameters", "what"]].round(4))

import matplotlib.pyplot as plt
t = table[table["experiment"] != "baseline"].sort_values("r2_gain")
fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.2))
y = np.arange(len(t))
ax.barh(y, t["r2_gain"], color=["#2a9d8f" if lo > 0 else "#e76f51" if hi < 0 else "#adb5bd"
                                 for lo, hi in zip(t["gain_ci_low"], t["gain_ci_high"])])
ax.errorbar(t["r2_gain"], y, xerr=[t["r2_gain"] - t["gain_ci_low"], t["gain_ci_high"] - t["r2_gain"]], fmt="none", ecolor="k", capsize=3, lw=1)
ax.set_yticks(y, [f"{e} (seed {s})" for e, s in zip(t["experiment"], t["seed"])])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("test R² gain over baseline (95 % station-bootstrap interval)")
ax.set_title("Amouei elements: green = helps, red = hurts, grey = no clear effect")
fig.tight_layout()
fig.savefig(AM / "experiments" / "results" / f"{stem}_gain.png", dpi=150)
plt.show()
''')

md("""
## 6. Verdict
An element is worth adding to `model/` only if its gain is positive with the **whole** interval above 0 (and, with two seeds, in both).
""")
code(r'''
for exp, g in table[table["experiment"] != "baseline"].groupby("experiment"):
    sure = (g["gain_ci_low"] > 0).all()
    worse = (g["gain_ci_high"] < 0).all()
    verdict = "ADD to the model" if sure else "makes it WORSE" if worse else "no clear effect"
    print(f"{exp:20s} gain {g['r2_gain'].mean():+.4f}  [{g['gain_ci_low'].min():+.4f}, {g['gain_ci_high'].max():+.4f}]  -> {verdict}")
if ON_KAGGLE:
    shutil.copytree(AM / "experiments" / "results", Path("/kaggle/working") / "amouei_results", dirs_exist_ok=True)
    print("results also copied to /kaggle/working/amouei_results")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    src = c["source"]
    lines = src.split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
