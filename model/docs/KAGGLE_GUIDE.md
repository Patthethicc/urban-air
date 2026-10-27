# Training the CNN-LSTM on Kaggle: the guide

Everything below was first run end to end on this repo in a local stand-in for Kaggle (a fake `/kaggle/input`, a fake
`/kaggle/working`, the same cells), at tiny scale. Section 11 says exactly what that proved and what only a real Kaggle run can
prove. Read it before you trust a number.

**The shape of the work.** Code lives on GitHub, data lives in a private Kaggle Dataset, every run lives in the notebook's
`/kaggle/working`. The notebook is a thin runner: it clones the code, copies the data next to it, and calls the same
`run_pipeline.py` commands you use locally. You never edit code in the notebook.

| What | Where | Why |
|---|---|---|
| Code | GitHub (`Patthethicc/urban-air`), cloned into `/kaggle/working/urban-air` | one source of truth; `git clone` brings only tracked files (about 7 MB) |
| Data | a private Kaggle Dataset (about 1 GB), copied into `/kaggle/working/urban-air/data` | `data/` is git-ignored, so a clone does not have it |
| Runs | `/kaggle/working/urban-air/model/runs/<dataset>/{processed,checkpoints,snapshots,tuning}` | only `/kaggle/working` survives a session, and only after a committed run |

For the experiment itself (which datasets, which seeds, how to choose, when to tune) follow `docs/EXPERIMENT_CHECKLIST.md`. This
guide is the how-to-run-it-on-Kaggle half.

## 1. Platform facts, and how sure I am of them

| Fact | Source | Status |
|---|---|---|
| Sessions last up to 12 h (CPU and GPU), 9 h (TPU) | [Kaggle docs](https://www.kaggle.com/docs/notebooks), via search | confirm in the notebook's panel |
| `/kaggle/working` keeps 20 GB; there is separate scratch space that is not saved | [Kaggle docs](https://www.kaggle.com/docs/notebooks), via search | confirm |
| CPU session: 4 cores, 30 GB RAM; GPU sessions: P100 (1 GPU) or T4 x2 (2 GPUs), 4 cores, 29 GB RAM | [Kaggle docs](https://www.kaggle.com/docs/notebooks), [product update](https://www.kaggle.com/product-feedback/361104) | confirm |
| GPU quota about 30 h per week, resets weekly, "or sometimes higher"; P100 and T4 share it | [GPU usage tips](https://www.kaggle.com/docs/efficient-gpu-usage) | confirm |
| GPU and Internet need a phone-verified account; Internet is a toggle in the notebook settings | [fast.ai thread](https://forums.fast.ai/t/beginner-using-colab-or-kaggle/96280) | confirm |
| Private dataset quota 200 GB (older posts say 20 GB per dataset). Our upload is 984 MB, so either is fine | [announcement](https://www.kaggle.com/product-announcements/512322) | irrelevant at this size |
| A forum post quotes a limit of 50 TOP-LEVEL files per dataset. Our upload has 4 top-level folders | [forum](https://www.kaggle.com/general/216170) | we are far inside it, but keep the folder structure |
| Idle timeout of interactive sessions | no current source found | unknown: use "Save & Run All" for anything long |
| Python / pandas / numpy / torch versions of the Kaggle image | not found | unknown: Section 4 cell 5 (the test suite) is the check |

Our own dev environment is Python 3.14, pandas 3, numpy 2.4, torch 2.13. Kaggle's image will differ.

## 2. Before you open Kaggle (local, once)

1. **Commit and push the code.** The newest work (pinned CV folds, `compare_runs`, the checklist, this guide) was not committed when this
   guide was written, and a clone only has what is pushed.
   ```bash
   git status
   ```
   `data/` must not be listed as something to add (the root `.gitignore` ignores it; commit that file too so it is true for everyone).
   ```bash
   git add model .gitignore
   ```
   ```bash
   git commit -m "feat: pinned CV folds, run comparison tool, experiment checklist, Kaggle guide"
   ```
   ```bash
   git push
   ```
2. **Run the tests locally**: `cd model && python -m pytest -q` should end `416 passed, 4 skipped`.
3. **Build the upload folder** (it copies only what the pipeline reads: 984 MB, 607 files, 4 top-level folders; it leaves out the
   `combined_hourly.csv` files, which the pipeline never reads, and `QC Data/`, which only the converter needs). Run from the repo root:
   ```python
   import shutil
   from pathlib import Path
   src, out = Path("data"), Path("kaggle_upload/urban-air-data")
   arms = ["full", "from_2023", "from_2024", "full_with_preinstall", "from_2023_with_preinstall", "from_2024_with_preinstall"]
   for a in arms:
       base = src / "Combined Dataset" / a
       shutil.copytree(base / "sensors", out / "Combined Dataset" / a / "sensors")
       shutil.copy2(base / "locations.csv", out / "Combined Dataset" / a / "locations.csv")
   for f in ("README.md", "stations.csv", "excluded_readings.csv"):
       shutil.copy2(src / "Combined Dataset" / f, out / "Combined Dataset" / f)
   shutil.copytree(src / "OpenAQ", out / "OpenAQ")                       # the `raw` arm
   shutil.copytree(src / "QC Data Converted", out / "QC Data Converted")  # the `raw` arm
   if (src / "Boundaries" / "ncr_cities.gpkg").is_file():                  # the NCR city outlines for the maps (optional: without it the maps are unclipped)
       (out / "Boundaries").mkdir()
       shutil.copy2(src / "Boundaries" / "ncr_cities.gpkg", out / "Boundaries" / "ncr_cities.gpkg")
   (out / "Morphology").mkdir()
   shutil.copy2(src / "Morphology" / "master_raster_v2.tif", out / "Morphology" / "master_raster_v2.tif")
   ```
4. **Upload it.** On kaggle.com: Create, New Dataset, drag in the `urban-air-data` folder, set the title to `urban-air-data`, **Private**,
   Create. Check that you are allowed to share the Quezon City sensor data. If the upload misbehaves with so many small files, zip the
   folder instead. Kaggle unpacks a zip as it is, so a zipped `data` folder arrives as `/kaggle/input/urban-air-data/data/Morphology/...`
   (one folder deeper than the unzipped upload); Cell 4 finds the raster either way.

## 3. Create the notebook

New Notebook, then in the right-hand panel:

- **Add Input**: your `urban-air-data` dataset. Note the path it shows; it should be `/kaggle/input/urban-air-data`.
- **Session options**: accelerator **GPU P100** (one GPU; the code uses one device, so a T4 x2 session leaves the second idle),
  **Internet On** (needed for `git clone` and `pip`).
- If the repo is private: Add-ons, Secrets, add `GITHUB_TOKEN` (a read-only personal access token) and attach it to the notebook.

## 4. The cells

Run them in order. Cells 1 to 6 are setup (about 4 minutes on CPU, mostly the tests). Everything after is one decision at a time.

**Cell 1: is the GPU there?**
```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU: turn the accelerator on and restart")
```

**Cell 2: paths and the runner.** This cell is the whole interface to the pipeline. Re-run it after any restart.
```python
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK  = Path("/kaggle/working")
DATA  = Path("/kaggle/input/urban-air-data")          # the exact path shown in the Input panel
REPO  = WORK / "urban-air"
MODEL = REPO / "model"
ARMS  = ["raw", "full", "from_2023", "from_2024", "full_with_preinstall", "from_2023_with_preinstall", "from_2024_with_preinstall"]
PREFER = ["full", "from_2023", "from_2024", "full_with_preinstall", "from_2023_with_preinstall", "from_2024_with_preinstall", "raw"]   # tie-break order (checklist D3)

def overrides(arm, extra=()):
    """Every stage of an arm needs the same path overrides, so one function builds them. `raw` keeps the sensor paths of data.yaml."""
    dirs = {f"data.paths.{k}_dir": f"runs/{arm}/{k}" for k in ("processed", "checkpoints", "snapshots", "tuning")}
    paths = {} if arm == "raw" else {"data.paths.sensor_locations_csv": f"../data/Combined Dataset/{arm}/locations.csv",
                                      "data.paths.sensor_csv_dir": f"../data/Combined Dataset/{arm}/sensors"}
    kv = {"data.split.cv_assignment": "by_station", **paths, **dirs}          # by_station: the same CV folds for every dataset
    return [x for k, v in kv.items() for x in ("--override", f"{k}={v}")] + list(extra)

def run(stage, arm, *flags, extra=()):
    cmd = [sys.executable, "-u", "run_pipeline.py", "--stage", stage, *overrides(arm, extra), *flags]
    print("$", " ".join(cmd), flush=True)
    t = time.time()
    subprocess.run(cmd, cwd=MODEL, check=True)                                  # stops the notebook if a stage fails
    print(f"-- {stage} {arm}: {time.time() - t:.0f} s", flush=True)

def runs_of(arm):
    """Finished training runs of an arm (run_info.json says completed)."""
    out = []
    for info in sorted((MODEL / "runs" / arm / "checkpoints").glob("*/run_info.json")):
        if json.loads(info.read_text(encoding="utf-8")).get("status") == "completed":
            out.append(info.parent.name)
    return out

def seed_args(seed):
    return ["--override", f"train.seed={seed}"]

def compare():
    subprocess.run([sys.executable, "-m", "reporting.compare_runs", "--runs-root", "runs", "--out-dir", "runs/comparison",
                    "--prefer", ",".join(PREFER)], cwd=MODEL, check=True)          # prints the table and writes runs/comparison/
```

**Cell 3: get the code.** Public repo:
```python
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Patthethicc/urban-air.git", str(REPO)], check=True)
```
Private repo (the token never reaches the output, and is removed from the clone's settings afterwards):
```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
r = subprocess.run(["git", "clone", "--depth", "1", f"https://{token}@github.com/Patthethicc/urban-air.git", str(REPO)], capture_output=True, text=True)
if r.returncode:
    raise SystemExit("clone failed: " + r.stderr.replace(token, "***"))
subprocess.run(["git", "-C", str(REPO), "remote", "set-url", "origin", "https://github.com/Patthethicc/urban-air.git"], check=True)
```

**Cell 4: put the data next to the code** (15 s; 984 MB). The pipeline reads `../data/...` relative to `model/`, so `Morphology/master_raster_v2.tif`
must end up at `/kaggle/working/urban-air/data/Morphology/`. The cell looks for that raster under `DATA`, so it works whether the upload is the
folder itself, a zip that Kaggle unpacked (a zipped `data` folder arrives one level deeper), or a zip it left packed, and it copies only the four folders the pipeline reads.
```python
rasters = list(DATA.rglob("master_raster_v2.tif"))
if rasters:
    src = rasters[0].parent.parent                            # the folder that holds Morphology/, OpenAQ/, ...
    for name in ("Combined Dataset", "Morphology", "OpenAQ", "QC Data Converted", "Boundaries"):
        if (src / name).is_dir():                                 # Boundaries is optional: without it the maps are not clipped to the NCR
            shutil.copytree(src / name, REPO / "data" / name, dirs_exist_ok=True)
    print("data copied from", src)
else:                                                         # Kaggle did not unpack an uploaded zip: its top-level `data/` folder lands in REPO
    import zipfile
    zips = list(DATA.rglob("*.zip"))
    assert zips, f"no master_raster_v2.tif and no zip under {DATA}: attach the dataset in the Input panel and set DATA to the path shown there"
    zipfile.ZipFile(zips[0]).extractall(REPO)
    print("data unzipped from", zips[0])
assert (REPO / "data/Morphology/master_raster_v2.tif").is_file(), f"the raster is not at {REPO}/data/Morphology: look at `ls {REPO}/data`"
print(sorted(p.name for p in (REPO / "data").iterdir()))     # ['Boundaries', 'Combined Dataset', 'Morphology', 'OpenAQ', 'QC Data Converted'] (plus 'QC Data' if you unzipped)
```
Do not copy `DATA` as a whole: a zipped `data` folder would land as `data/data/...` and everything that reads the raster fails.

**Cell 5: install only what is missing, then run the test suite.** The suite is the compatibility check for Kaggle's Python and library versions.
It needs the data in place (about 20 tests read the raster), and Kaggle's preinstalled `esda` can be older than the tested 2.10.0 (its `Moran_Local`
then lacks the `alternative` option the spatial-bias check uses), so this cell stops without data and upgrades an old `esda`.
```python
import importlib.util
assert (REPO / "data/Morphology/master_raster_v2.tif").is_file(), "the data is not in place: run Cell 4 first (the tests and the pipeline read it)"
need = {"skopt": "scikit-optimize", "libpysal": "libpysal", "esda": "esda", "rasterio": "rasterio", "geopandas": "geopandas",
        "shapely": "shapely", "pyproj": "pyproj", "affine": "affine", "yaml": "pyyaml", "sklearn": "scikit-learn", "pytest": "pytest"}
missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
print("installing:", missing or "nothing")
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
check_esda = "import inspect; from esda.moran import Moran_Local as M; raise SystemExit(0 if 'alternative' in inspect.signature(M.__init__).parameters else 1)"
if subprocess.run([sys.executable, "-c", check_esda]).returncode != 0:          # run in its own process: nothing is imported here
    print("esda is too old (no two-sided LISA option): upgrading")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "esda", "libpysal"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=MODEL)
```
Expected: no failures, about `416 passed` after a `git clone` (the number of skipped tests depends on which optional libraries Kaggle has). If you uploaded the repo as a dataset instead (no `.git` folder), exactly one test fails,
`test_dotenv_is_git_ignored_but_the_template_is_not`, because it asks git a question. That one is harmless. Any other failure means
a library version differs: read the message before spending GPU hours.

**Cell 6: the smoke run, and your GPU timing.** One dataset, 1 epoch, large batch: it exercises every training step and tells you
the speed. On this repo's CPU (4 threads) this cell took 21 s + 503 s.
```python
t = time.time()
run("preprocess", "from_2024")
run("train", "from_2024", extra=["--override", "train.epochs=1", "--override", "train.batch_size=1024"])
print(f"smoke run: {time.time() - t:.0f} s")
```
Look at the output of `train`: on a GPU, `[fold 0] epoch 1/1 ...` should come far sooner than the CPU's first-epoch estimate. If the GPU sits
idle (check `!nvidia-smi` in another cell while it runs), the Python data loader is the limit: raise `train.batch_size`, or add
`--override train.num_workers=2`. Both are untested on a real GPU, so measure rather than assume.

Then work out your budget. One training costs about (K + K/(K-1)) x epochs x seconds-per-epoch with K = 3 folds (3 CV folds on two-thirds of
the stations, then a final fit on all of them), and early stopping usually ends each fit after roughly 5 to 12 epochs.

| Planned | Trainings | Your seconds per training | Hours |
|---|---|---|---|
| Stage A: 7 datasets x seed 42 | 7 | | |
| Stage B: best 2 to 3 x seeds 43, 44 | 4 to 6 | | |
| Tune (one dataset) | see Cell 9 | | |
| Final: tuned x 3 seeds + untuned x 3 seeds | 3 (the untuned ones are already in Stage B) | | |

Keep every committed session under about 10 hours; there is no resume inside a stage.

**Cell 7: Stage A, screening.** One seed per dataset, `preprocess` and `train` only (no `geocp`, no `report`, so no test numbers exist). It
skips datasets that already have a finished run, so after a restart you can simply run it again.
```python
for arm in ARMS:
    if runs_of(arm):
        print("skip", arm, "(already trained)")
        continue
    run("preprocess", arm)
    run("train", arm, extra=seed_args(42))
compare()
```
If you do not want to spend a whole session on this, run `ARMS[:3]` in one session and the rest in the next.

**Cell 8: Stage B, confirm.** Fill in the datasets that came out best (or tied best), then repeat with two more seeds.
```python
BEST = ["from_2023", "from_2024"]            # your choice from the Stage A table
for arm in BEST:
    for seed in (43, 44):
        if len(runs_of(arm)) < {43: 2, 44: 3}[seed]:
            run("train", arm, extra=seed_args(seed))
compare()
```
Read `runs/comparison/comparison.md` against the checklist's rule. **Pick the dataset by that table, not by any run's own `SUMMARY.md`** (those cover different hours,
and no training-stage run has read the test partition yet). Write the choice down before the next cell.

**Cell 9: tune the chosen dataset.** Own session. The search cannot resume, so give it a budget that fits.
```python
CHOSEN = "from_2023"                          # decided in Stage B
run("tune", CHOSEN, extra=["--override", "tune.n_calls=12", "--override", "tune.row_stride=8"])
print(sorted((MODEL / f"runs/{CHOSEN}/tuning").glob("*/best_hyperparameters.json")))
```
The default is a nested search (3 outer folds, 3 inner, 12 trials per search), about 20 to 25 hours on this repo's CPU. Cheaper:
`--override tune.nested_evaluation=false` (only the final search, no outer-fold estimate) and/or a larger `tune.row_stride`. The stand-in ran a tiny version
(`nested_evaluation=false`, 2 trials, `row_stride=400`) in 23 s to prove the command works. On a GPU the first trial prints its duration and a
projection for all searches; stop early if the projection does not fit.

**Cell 10: the final models, and the single test evaluation.** Tuned (three seeds) next to the untuned runs you already have.
```python
for seed in (42, 43, 44):
    run("train", CHOSEN, "--tuned", extra=seed_args(seed))
import pandas as pd
index = pd.read_csv(MODEL / f"runs/{CHOSEN}/checkpoints/runs_index.csv")
print(index[["run_id", "tuned", "batch_size", "cv_rmse_mean", "final_epochs"]].to_string(index=False))
```
This is the first time the test partition is read. With `TUNED` and `UNTUNED` set to run ids from that table, score both on the same test rows:
```python
TUNED, UNTUNED = "20260101_000000", "20260101_010000"      # replace with real run ids
run("report", CHOSEN, "--run-id", TUNED, "--baseline", UNTUNED)
print((MODEL / f"runs/{CHOSEN}/snapshots/{TUNED}/SUMMARY.md").read_text(encoding="utf-8")[:3000])
```
`report` runs `geocp` and the spatial-bias check first if they are missing. Expect a small gain from tuning (the pilot gave none).

**Cell 11: the maps and Grad-CAM.** A 2025 date uses every training station. The Quezon City sensors stopped on 2025-12-31, so for dates from 2026 on
`infer` and `explain` build the history from the nearest OpenAQ training stations only (`inference.openaq_only_from_year: 2026`, printed when `infer` starts);
without that, about 12 % of the cells would get no prediction. Add `"--override", "inference.openaq_only_from_year=null"` to `extra` to switch it off.
```python
run("infer", CHOSEN, "--run-id", TUNED, "--date", "2025-07-01", extra=["--override", "inference.stride_px=11"])
run("explain", CHOSEN, "--run-id", TUNED, "--timestamp", "2025-07-01 12:00", extra=["--override", "inference.stride_px=11"])
```
The web basemap needs Internet and a CARTO key; without them the maps use the offline basemap (street lines) or a plain background, and the
numbers are the same. Add `"--override", "output.basemap.enabled=false"` to skip the attempt. `model/.env` is git-ignored, so a clone has none:
the pipeline reads `CARTO_API_KEY` from the environment instead (a real environment variable wins over `.env`). Put the key in Kaggle Secrets
(Add-ons, Secrets, label `CARTO_API_KEY`, switched on for this notebook) and run this **right before the maps, not in Cell 5**: the test suite is
meant to run without real keys. The `run()` helper's subprocess inherits it. Never print the key.
```python
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["CARTO_API_KEY"] = UserSecretsClient().get_secret("CARTO_API_KEY")
    print("CARTO_API_KEY set from Kaggle Secrets")
except Exception as e:          # not attached to this notebook, a different label, or Secrets unavailable
    print(f"CARTO_API_KEY not set ({type(e).__name__}): the maps will use the offline basemap")
```
Stride 11 is the quick look; stride 1 is the full 10 m map and takes hours on a CPU. Measure it on the GPU first.

**Cell 12: tidy up and save.** Everything under `/kaggle/working` is the notebook's output when the run is committed.
```python
def mb(p): return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 2**20
print({d.name: round(mb(d)) for d in (MODEL / "runs").iterdir()}, "MB")
for arm in ARMS:                              # processed/ is 100 MB per dataset and `preprocess` rebuilds it in about 20 s
    if arm != CHOSEN and (MODEL / "runs" / arm / "processed").exists():
        shutil.rmtree(MODEL / "runs" / arm / "processed")
```

## 5. What the stand-in run produced (so you know what "working" looks like)

One dataset (`from_2024`), CV on, 1 epoch, batch 1024, on a 4-thread CPU, with every stage run through the Cell 2 runner:

| Stage | Time | Evidence it worked |
|---|---|---|
| copy data | 15 s | `['Combined Dataset', 'Morphology', 'OpenAQ', 'QC Data Converted']` |
| tests | 166 s | 364 passed, 1 expected failure (no `.git` in the stand-in) |
| preprocess | 21 s | 54 / 13 / 17 stations, 712,172 / 190,820 / 239,097 rows |
| train | 503 s | 3 folds + final fit; `TRAINING SUMMARY` printed |
| geocp | 30 s | test coverage 0.912 against the 0.90 target |
| report | 13 s | `SUMMARY.md`, plots, tables |
| infer, stride 21, one date | 45 s | 4 snapshots, GeoTIFF volumes, 6 PNG maps |
| explain | 11 s | Grad-CAM hot and cold cells |
| compare_runs | 3 s | `comparison.md` and two CSVs |
| tune, tiny | 23 s | `best_hyperparameters.json`, `tune_info.json` |

Space for that one dataset: `processed` 101 MB, `checkpoints` 49 MB, `snapshots` 78 MB: about 230 MB, so seven datasets fit comfortably in 20 GB.
The result of a 1-epoch model is meaningless as accuracy; it only proves the plumbing.

## 6. The folder you end up with
```
runs/
  comparison/                    comparison.md, comparison_arms.csv, comparison_runs.csv
  <dataset>/
    processed/                   the model's input arrays (rebuildable with `preprocess`)
    checkpoints/<run_id>/        best_dual_branch_model.pt, fold_*.pt, history.json, oof_predictions.csv, run_info.json
    checkpoints/runs_index.csv   one row per run: settings and scores
    snapshots/<run_id>/          SUMMARY.md (start here), plots/, tables/, data/, surface/, explain/
    tuning/<id>/                 best_hyperparameters.json, tune_info.json, search_*.json
```

## 7. Sessions: how to split the work, and how to keep it
- **Only `/kaggle/working` is kept, and only when you press "Save Version" (Save & Run All).** Interactive sessions can stop on idle, so run anything
  long as a committed run, and put the heavy stage in its own notebook version.
- **No resume inside a stage.** A killed `train` leaves a run folder whose `run_info.json` says `running`; the runner's `runs_of()` ignores it, and the
  comparison tool skips it. Re-run just that dataset.
- **Chaining sessions:** a later notebook can take an earlier notebook's output as an input. Look for it with
  `!find /kaggle/input -maxdepth 4 -name runs`, then copy it back:
  ```python
  shutil.copytree("<path found above>", MODEL / "runs", dirs_exist_ok=True)
  ```
  After that `runs_of()` and `compare()` see the earlier work. I could not test this part here: confirm the path on Kaggle.
- **pip installs and the cloned code are not kept** between sessions; Cells 2 to 5 rebuild them in about a minute.

## 8. Things the data will do (known, not bugs)
- **Quezon City sensors end on 2025-12-31.** The spatial lag takes each cell's five nearest training stations whether or not they report, which would leave holes
  over Quezon City on 2026 dates (88.3 % of cells predicted on 2026-04-11, 100 % on 2025-07-01). `inference.openaq_only_from_year: 2026` (the default) avoids that for
  `infer` and `explain` by using the nearest OpenAQ stations from 2026 on; the history is then farther away and made of one network. `SUMMARY.md` section 5 shows
  the test accuracy per year, so you can see how 2026 (OpenAQ test stations only) compares with the years that include Quezon City.
- **One test station contributes nothing.** OpenAQ station 6340868 reports only from 2026-05, and its five nearest training stations are all Quezon City sensors, so
  none of its rows can be built. The effective test set is 16 stations in every dataset, not 17 (about 1 % of the valid hours are lost this way; only this station is
  affected entirely).
- **The two networks read differently** (median hourly network mean 13.2 against 17.2 µg/m3 where both operate) and cannot be calibrated against each other.
- **GPU runs are not bit-identical to CPU runs.** Compare datasets only against runs made on the same hardware.

## 9. If something goes wrong
| You see | Likely cause and what to do |
|---|---|
| `torch.cuda.is_available()` is False | the accelerator is off: Session options, GPU P100, then restart |
| `git clone` fails | Internet is off, or the repo is private (use the token variant) |
| `ModuleNotFoundError: preprocessing` | the command did not run with `cwd=MODEL`; use the `run()` helper |
| `FileNotFoundError ... locations.csv` or `master_raster_v2.tif` | Cell 4 did not copy the data, or `DATA` points at the wrong folder |
| Cell 5 shows about 20 failures, all mentioning `master_raster_v2.tif` (`test_combine_dataset`, `test_inference_reuse`, `test_coverage`, `test_crs`) | the same cause: the raster is not at `/kaggle/working/urban-air/data/Morphology/`. Most often the upload was a zipped `data` folder and was copied one level too deep (`data/data/Morphology`). Use the Cell 4 above, which finds the raster itself; check with `ls /kaggle/working/urban-air/data` |
| `esda ... is too old for the two-sided LISA p-values` (or `Moran_Local ... unexpected keyword argument 'alternative'` in the tests) | Kaggle's preinstalled esda is older than the tested 2.10.0: `pip install -U esda libpysal`, restart the session, re-run Cells 2 to 5 |
| `CARTO basemap tiles need a free API key` on Kaggle | `CARTO_API_KEY` is not in the environment (`model/.env` is not cloned). Re-run the Secrets lines of Cell 11 (they are lost on a restart); check the secret is attached to this notebook and spelled exactly `CARTO_API_KEY`; web tiles also need Internet on |
| `test_the_real_xyzservices_carto_provider_is_recognised_and_keyed` is skipped | Kaggle's xyzservices is older and its CARTO template has no key placeholder; the code adds `?key=` itself (`test_a_carto_template_without_a_key_placeholder_gets_the_key_appended`), so results are not affected |
| `ModuleNotFoundError: skopt` (or `libpysal`, `esda`) | Cell 5 was skipped; `tune` needs scikit-optimize, `report` needs libpysal and esda |
| `CUDA out of memory` in `train` | lower `train.batch_size`; in `infer` lower `inference.chunk_cells` (default 1024) |
| `TIMEOUT in stage ... no progress for 600 s` (exit code 124) | a real hang was detected; see `runtime:` in `configs/train.yaml`; re-run that stage |
| `checkpoint does not match the processed data` | you re-ran `preprocess` with different settings after training; train again |
| Output is not saved | the run was not committed ("Save Version"), or it failed; only a successful committed run keeps `/kaggle/working` |
| The session ends mid-stage | no resume: re-run the unfinished dataset (`runs_of()` shows what is done) |
| `compare_runs` says "no common rows" | the runs cover disjoint periods or stations; compare only datasets that overlap |

## 10. After Kaggle
Download what you need from the notebook's Output tab: `runs/comparison/`, the final run's `snapshots/<run_id>/` (with `SUMMARY.md`), the final
`checkpoints/<run_id>/best_dual_branch_model.pt` and `run_info.json`, and `tuning/<id>/best_hyperparameters.json`. To run `infer` or `explain` locally on a Kaggle-trained model,
you also need that dataset's `processed/` folder (rebuild it locally with the same overrides, or download it), because the checkpoint is checked against it.

## 11. What the stand-in proved, and what it did not
**Proved (run locally with the same cells):** the upload folder builds (984 MB, 607 files); the data copy; the test suite runs from the uploaded layout; `preprocess`,
`train`, `geocp`, `report`, `infer`, `explain`, `compare_runs` and `tune` all complete through the Cell 2 runner; the by-station CV folds are used;
the device handling code moves every tensor to the device before use and back to the CPU before `.numpy()` (read, not run).

**Not proved, because it needs Kaggle or a GPU:** anything on a real GPU (this machine has only a CPU build of PyTorch: speed, memory, the data loader
keeping the GPU busy, `torch.unique` on CUDA); Kaggle's Python and library versions (Cell 5 tells you); `git clone` from your GitHub; Kaggle Secrets;
the dataset mount path and upload behaviour; committed-run persistence and chaining sessions; quotas and the idle timeout.

If a step here disagrees with what Kaggle shows you, trust Kaggle and tell me; the cells are small enough to adjust.
