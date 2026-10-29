# XGBoost branch — street-level PM2.5 from urban morphology

Street-level PM2.5 estimation from urban morphology and temporal features, using XGBoost with nested
spatial cross-validation, Bayesian hyperparameter optimisation, a three-way ablation, geographically
weighted conformal prediction intervals, and city-wide inference.

**Two datasets, one pipeline.** The branch now runs on either, selected by one environment variable:

| | `XGB_DATASET=pilot` | `XGB_DATASET=full` *(default)* |
|---|---|---|
| Locations | 18 sensors / 17 cells | **84 locations / 83 cells** |
| Rows kept | 43,273 (of 122,841) | **1,256,692** (of 1,487,811) |
| Grid | 378 cells, 10.5 × 9 km | **3,696 cells**, ~24 × 39 km |
| Split | 11 train / 3 calibration / 3 test | **64 / 12 / 7** |
| Status | the published pilot record | the real-world run |

**Scope: this document covers the XGBoost branch only.** Everything lives in `XGBoost/`. `CNN-LSTM/` is
**not** part of this branch and is not documented here — it is kept as a **worked example of the intended
output** (stages, configs, tests, snapshot surfaces, AQI maps) and as the source of two techniques
borrowed here: the stand-in history for unmonitored cells and the raster/extent recipe.

---

## Repository structure

The repository root **is** this branch — paths below are relative to it. It sits in `xgboost/` of the
thesis repository, beside the CNN-LSTM (`model/`) and the pilot record, but nothing outside this folder is
needed to run anything here. Only source code and documentation are versioned: **48 files, 0.47 MB**.

```
xgboost/                          this branch  — 48 files, 0.47 MB
├── README.md                   38.1 KB  this file
├── GUIDE.md                     7.5 KB  cell-by-cell guide + glossary
├── .env.example                 0.5 KB  XGB_DATASET / XGB_FOLDS / cleaning switches
├── requirements.txt             0.8 KB  pinned versions
├── data.py                      7.2 KB  resolves every path; reads .env
├── run_pipeline.py             15.0 KB  staged entry point
├── build_morphology_grid.py     6.9 KB  10 m raster -> 500 m grid + sensor_locations.csv
├── ingest_sources.py           12.8 KB  raw per-sensor CSVs -> Master_DL_Ready_full.csv
├── xgboost_full.ipynb          32.8 KB  the notebook, local and Kaggle
│
├── preprocessing/
│   ├── __init__.py
│   ├── prep.py                 23.6 KB  merge -> EDA cleaning -> lags -> features -> spatial lag
│   ├── spatial_cv.py           21.5 KB  k-means spatial folds, partitions, JSON persistence
│   ├── grid_builder.py          9.6 KB  vector grid + zonal statistics
│   ├── patch_stats.py           6.0 KB  10 m patch statistics for the TEXTURE arm
│   └── spatial_bias.py          4.2 KB  Moran's I / LISA on out-of-fold residuals
│
├── training/
│   ├── __init__.py
│   ├── ablation.py             20.0 KB  five arms + nested spatial CV + Bayesian search
│   ├── shared_split.py          7.4 KB  writes the 54/13/16 split (station ids are literals)
│   ├── partitions.py            9.1 KB  this branch's own split; refuses to clobber the shared one
│   ├── final_test.py            8.0 KB  scores the 16 reserved test stations
│   ├── artifacts.py             5.2 KB  model.joblib + run_info.json + features.json
│   └── ablation_outputs.py      1.8 KB  guard for the regenerable large outputs
│
├── uncertainty/
│   ├── __init__.py
│   ├── geocp.py                39.6 KB  GeoCP, split conformal, bandwidth search, batched quantiles
│   └── geocp_shared.py          8.2 KB  the CNN-LSTM-matched kernel x bandwidth search
│
├── inference/
│   ├── __init__.py
│   └── cell_inference.py       14.7 KB  per-cell stand-in history at one hour
│
├── maps/
│   ├── __init__.py
│   ├── basemap.py              12.6 KB  OSM tiles, keyless; cache in .tile_cache or data/
│   └── surfaces.py             25.5 KB  per-snapshot maps, 2x2 panel, 24-hour mean
│
├── reporting/
│   ├── __init__.py
│   ├── dataset_report.py       18.4 KB  Deliverable 1
│   └── figures.py              18.8 KB  the nine report figures, including SHAP
│
├── experiments/                        each backs a number quoted in this README
│   ├── target_transform.py      4.7 KB  log1p vs raw vs standardised
│   ├── persistence_by_fold.py   3.3 KB  model vs 1-hour persistence, per fold
│   ├── eda_cleaning_impact.py   3.1 KB  paired A/B of the EDA cleaning
│   └── cohort_window.py         3.8 KB  2024-01+ cohort-aware sensitivity
│
└── checks/                             self-tests; none is imported by the pipeline
    ├── __init__.py
    ├── run_notebook.py          2.5 KB  executes the notebook through nbconvert
    ├── run_notebook_inprocess.py 4.3 KB  executes it in-process, cell by cell
    ├── verify_shared_split.py   6.7 KB  the 54/13/16 partition is reproducible
    ├── verify_ablation.py      10.3 KB  ablation numbers against the saved predictions
    ├── verify_geocp.py         10.4 KB  GeoCP coverage and width
    ├── test_notebook_flow.py    6.8 KB  every stage the notebook calls
    ├── test_spatial_lag.py      5.7 KB  the fold-aware spatial lag
    ├── check_morphology_coverage.py 3.0 KB
    ├── check_output_coherence.py    8.8 KB
    └── adversarial_step3.py    15.5 KB```

`data/` and `outputs/` are **not versioned** — every file in them is either a raw input or written by a
stage. They exist only in a working copy:

```
xgboost/                           (only in a working copy, never committed)
├── data/                          inputs + stage-written tables
│   ├── OpenAQ/                    45 raw CSVs                          39 MB
│   ├── QC Data Converted/         41 raw CSVs                         112 MB
│   ├── master_raster_v2.tif       10 m morphology, 5 bands             87 MB
│   ├── Master_DL_Ready_full.csv   1.49 M rows, 84 sensors             171 MB  <- ingest_sources.py
│   ├── morphology_grid_full.csv   3,696 cells @ 500 m                0.8 MB  <- build_morphology_grid.py
│   ├── patch_stats.csv            35 texture features per cell        45 KB   <- --stage patch
│   ├── fold_partitions_shared.json 54 / 13 / 16 cells                  6 KB   <- --stage split
│   ├── sensor_locations.csv       84 stations                          5 KB
│   ├── imputed_cells.json / imputed_texture_cells.json / ingest_report.json
│   └── 20261016_070432/           the CNN-LSTM reference run (quoted, not used)
│
├── outputs/                       every file is generated
│   ├── ablation_{results,summary}.csv       15 rows: 5 arms x 3 folds
│   ├── test_metrics_all.csv                 one row per scored arm
│   ├── test_metrics_<arm>.csv, predictions_test_<arm>.csv
│   ├── oof_predictions_<arm>.csv            ~100 MB per arm
│   ├── dataset_report.md                    Deliverable 1
│   ├── figures/                             9 figures + arm_comparison(.png/_bars.png)
│   ├── geocp_shared/                        geocp.json, candidates, test intervals
│   ├── surfaces/<date>/                     per-hour CSV + GeoTIFF + AQI map + continuous map,
│   │                                        + panel_<date>.png + <date>_mean_continuous.png
│   └── artifacts/latest/                    model.joblib, run_info.json, features.json
│
└── .tile_cache/                   optional: 9 OSM tiles, 254 KB, makes maps render offline
```

Every writer creates its own directory (`os.makedirs(..., exist_ok=True)`), so `outputs/` need not exist at
all — a fresh clone creates it on the first stage.

## The dataset is on Kaggle, and it can be shared

The inputs are published as a **Kaggle dataset**, `xgboost-data`. It is private by default; **anyone you
give the link to can use it**, and you can also add collaborators in *Settings → Collaborators*. Nothing in
the pipeline is tied to the owner, so a shared copy works unchanged.

Upload it as **one folder, `dataexample/`** (zip it, or drag the folder). Rename nothing except the tile
cache, which must be `.tile_cache`.

| Contents | Size | Needed? |
|---|---|---|
| `OpenAQ/` | 39 MB | yes |
| `QC Data Converted/` | 112 MB | yes |
| `Morphology/master_raster_v2.tif` | 87 MB | yes |
| `Combined Dataset/full_with_preinstall/locations.csv` | 6 KB | yes |
| `.tile_cache/` (9 tiles) | 254 KB | optional — the map basemap |
| `Master_DL_Ready_full.csv` | 171 MB | optional — skips a ~4 min build |
| `morphology_grid_full.csv` + `.json` | 0.8 MB | optional — skips the grid build |
| `patch_stats.csv`, `sensor_locations.csv`, `fold_partitions_shared.json` | 55 KB | optional |
| `imputed_cells.json`, `imputed_texture_cells.json`, `ingest_report.json` | 30 KB | optional |

The other five `Combined Dataset/*` variants (~2 GB extracted) and `QC Data/` are **not read by this
pipeline** — delete them before uploading and the archive drops from ~217 MB to ~90 MB.

**The code is not in the dataset.** It comes from this repository, which is why the notebook has a
`git clone` branch. In Kaggle:

1. **Add Input** → your `xgboost-data` dataset (or a copy someone shared with you).
2. Import `xgboost/xgboost_full.ipynb` from this repository.
3. **Accelerator: None (CPU)**, **Internet: On** — the clone needs it.
4. **Save & Run All (Commit)**.

Cell 0 then either copies the code out of an attached code dataset, or clones this repository and
`cd`s into `xgboost/`, links the bundle into `data/`, and runs the build stages only for the files the
bundle does not already carry. It prints `folds`, `master` and `grid` before anything else, so a wrong
attach fails immediately instead of two hours in.

Full training on Kaggle takes **~2.5–3 hours** for the five-arm search, plus ~15 minutes for the surfaces.
The local run skips training entirely when `outputs/ablation_results.csv` already exists.

## Prerequisites: what is NOT in the repository

`data/` is not versioned, because every file in it is either a raw input or written by a script. Supply the
raw inputs, then let the pipeline build the rest:

```bash
python build_morphology_grid.py        # -> sensor_locations.csv, morphology_grid_full.csv
python ingest_sources.py               # -> Master_DL_Ready_full.csv   (~4 min)
python run_pipeline.py --stage split   # -> fold_partitions_shared.json (54/13/16)
python run_pipeline.py --stage patch   # -> patch_stats.csv
```

Two dependencies worth knowing:

- `build_morphology_grid.py` needs the OpenAQ station table. Pass it explicitly from the Kaggle bundle:
  `--openaq-locations "<bundle>/Combined Dataset/full_with_preinstall/locations.csv"`. Without that flag
  it falls back to `../CNN-LSTM/pilot testing/...`, which is not in this repository.
- `fold_partitions_shared.json` is deterministic despite being excluded: the 16 test and 13 calibration
  station ids are literals in `training/shared_split.py`, adopted from the CNN-LSTM run. `--stage split`
  reproduces the exact partition.

### If `data/`, `outputs/`, `checks/` or `config/` were already pushed

Remove them from version control without deleting them locally:

```bash
git rm -r --cached data outputs checks config
git rm --cached dataset_report.md xgboost_pm25_pilot.ipynb
git commit -m "remove generated data, outputs and dev-only scripts from version control"
```

- `data/` and `outputs/` — generated or raw; ~1.9 GB, and `Master_DL_Ready_full.csv` is over GitHub's
  100 MB per-file limit.
- `checks/` — seven verification scripts; development aids, run from the working copy.
- `config/` — empty.
- `dataset_report.md` — written by `--stage report`.
- `xgboost_pm25_pilot.ipynb` — the pilot record; its numbers are in the Results section.

## How to run it — step by step

Everything below is the **console**. The notebook does the same work and now runs on either dataset —
see [Run it from the notebook](#run-it-from-the-notebook) — but the console is the reproducible path.

### Step 0 — once

```
cd XGBoost
pip install xgboost scikit-optimize scikit-learn pandas numpy matplotlib shap rasterio affine pyproj
```

Rebuild the inputs only when the raw sensor data or the morphology raster changes (both are already
built in this checkout):

```
python ingest_sources.py                 # 85 raw CSVs      -> Master_DL_Ready_full.csv      (~4 min)
python build_morphology_grid.py          # master_raster_v2 -> morphology_grid_full.csv      (~6 min)
```

### Step 1 — train (the main run)

This is the command that produces the thesis numbers. Five arms, 20 Bayesian trials each, on the station
split shared with the CNN-LSTM branch:

```powershell
cd C:\Users\delac\Downloads\thesis_model\XGBoost
$env:XGB_FOLDS = 'shared'                        # bash: export XGB_FOLDS=shared

python run_pipeline.py --stage split             # 54 / 13 / 16 station split   (~1 min)
python run_pipeline.py --stage patch             # 10 m patch statistics       (~1 min)
python run_pipeline.py --stage ablation --arms A B C C+ TEXTURE --n-trials 60 --row-stride 4
```

**~1.5–2.5 hours.** Measured on the shared split: ~14 s per trial, ~6 min per fold for arms A and C
(each fold fits ~30 cells / 412 k rows), so the five arms come to roughly two hours rather than the
earlier 7–9 h estimate, which was based on the larger 64-cell split. It checkpoints after every
fold, so a crash costs one fold.

| Arm | Features | What it adds | Purpose |
|---|---|---|---|
| **A** TIME_ONLY | 18 | cyclical time + PM2.5 history | the persistence benchmark |
| **B** MORPH_TIME | 15 | morphology + time, **no** PM2.5 history | can morphology alone predict? |
| **C** FULL | 24 | everything | the current model |
| **`C+`** FULL+spatial lag | 27 | fold-aware k=5 neighbour readings | the CNN-LSTM's extra input |
| **TEXTURE** | 59 | C + 10 m patch statistics (320 m window) | the CNN's patch information |

C+ computes its lag inside each outer fold from that fold's fit cells only, so a held-out cell never
contributes to its own feature.

**It prints a line per trial, so it is never silently working:**

```
=== arm A TIME_ONLY (18 features) [arm 1/5] ===
    3 outer fold(s): [0, 1, 2]
    fold 0: tuning (20 trials x 3 inner folds, 804,632 train rows) ...
      trial  1/20  inner_rmse 0.28476  best 0.28476  <- best     60s  eta 1144s
      ...
    outer fold 0: cells=[...] n=452060 RMSE=7.462 MAE=3.795 R2=0.600 | n_est=269 | fold took 1100s
  arm A done in 3300.0s | mean RMSE ... MAE ... R2 ...
```

| Line | Frequency |
|---|---|
| `trial N/20` on arms A, C, `C+`, TEXTURE | ~every 1 min |
| `trial N/20` on arm B | ~every 10–20 s (it early-stops almost immediately) |
| `outer fold N:` | every ~20 min — **15 in total** (3 per arm) |
| `arm X done` | 5 times |

If nothing new appears for more than ~3 minutes on a non-B arm, that is worth investigating.

### Step 2 — score and package the arm that won

`ablation_summary.csv` decides which arm to carry forward. **Check first, don't assume:**

```cmd
python -c "import pandas as pd; s=pd.read_csv('outputs/ablation_summary.csv'); print(s.sort_values('pooled_r2',ascending=False)[['arm','label','n_features','pooled_rmse','pooled_mae','pooled_r2']].to_string(index=False))"
```

This is the shell-agnostic form: it works in **cmd.exe** and PowerShell alike. (`Import-Csv` is a
PowerShell-only cmdlet and fails in cmd, which is the prompt most Windows users have.)

Then run these four, replacing `A` with whichever arm came out on top:

```powershell
python run_pipeline.py --stage test --test-arm A
python run_pipeline.py --stage geocp --geocp-profile shared --max-cal 6000 --max-eval 20000
python run_pipeline.py --stage save
python run_pipeline.py --stage figures --figures-arm A
```

`--stage save` and `--stage infer` **deploy the winning arm automatically**, read from
`ablation_summary.csv` (highest pooled R²). Override with `--save-arm C` if you need a specific one.
These two stages used to hardcode arm C, which meant the artifact and every city-wide surface were
produced by the second-best model.

| Command | Produces |
|---|---|
| `--stage test --test-arm A` | `test_metrics.csv` — headline accuracy on the 16 reserved test stations |
| `--stage geocp --geocp-profile shared ...` | `geocp_shared/` — intervals and coverage on their kernel/bandwidth grid |
| `--stage save` | `artifacts/latest/` — the deployable model + provenance |
| `--stage figures --figures-arm A` | `figures/` — the six report figures for that arm |

> **In the current run arm A (TIME_ONLY) won**, so the commands above use `A`. That matters for the
> write-up: the thesis's "full" model (C) is *not* the best one, and neither is the CNN-LSTM's extra
> spatial lag nor the 10 m patch texture. The best model on this data is built from the station's own
> recent history plus time of day.

#### Scoring more than one arm is safe — they do not overwrite each other

Each arm writes its own files, so you can score all five and compare:

| File | Behaviour |
|---|---|
| `test_metrics_<arm>.csv`, `predictions_test_<arm>.csv` | **per arm, kept** |
| **`test_metrics_all.csv`** | **one row per arm, sorted by R² — the comparison table** |
| `test_metrics.csv` / `predictions_test.csv` | the most recent run only (these feed `geocp` and `figures`) |

```powershell
python run_pipeline.py --stage test --test-arm C
python run_pipeline.py --stage test --test-arm TEXTURE
python run_pipeline.py --stage figures --figures-arm C

python -c "import pandas as pd; print(pd.read_csv('outputs/test_metrics_all.csv').to_string(index=False))" 
```

**Only `geocp` and `save` are single-model stages** — run those for the winning arm alone.

**Why bother scoring several:** an arm can win on pooled R² and still lose on the extremes.
`within_5` / `within_10` / `bias` show that, and the tail is exactly where the CNN-LSTM beat this
branch (their R² 0.702 vs our 0.659 came almost entirely from peaks above 120 µg/m³).

### Step 3 — switch dataset if needed

`XGB_DATASET` selects the inputs. It defaults to `full`:

```
XGB_DATASET=full  python run_pipeline.py --stage ablation     # 84 locations, 3,696-cell grid
XGB_DATASET=pilot python run_pipeline.py --stage ablation     # 18 sensors,   378-cell grid
```

On Windows PowerShell use `$env:XGB_DATASET='pilot'` first (they are separate processes). Remember
`outputs/` is **shared** — see the warning above — so whichever you ran last owns the committed numbers.

### Measured timings (this machine, 84 locations, 1.26 M training rows)

| Stage | Time | Notes |
|---|---|---|
| `ingest_sources.py` | ~4 min | 85 files, 1.49 M rows |
| `build_morphology_grid.py` | ~6 min | 3,696 cells × 5 bands of zonal statistics |
| `--stage report` | ~3 min | |
| `--stage partitions` | ~3 min | includes the leakage test |
| `--stage ablation --smoke` | ~6 min | 1 trial, 1 outer fold, subsampled rows |
| `--stage ablation --n-trials 8` | ~1 h | cheapest valid search |
| 5 arms × 20 trials, full rows | ~2.5 h | the original setting; ~14 s/trial, ~6 min/fold |
| 5 arms × 60 trials, `--row-stride 4` | **~1.9 h** | the current default: 3× the trials at ¼ the rows per trial, and the search actually converges |
| `--stage ablation` alone (3 arms) | ~2.5–4 h | measured: 1 arm × 8 trials × 3 folds took >25 min (~24 fits ≈ 50 s/fit) |
| `--stage geocp --max-cal 6000 --max-eval 20000` | ~5.5 min | caps required at this size |
| `--stage save` | ~1 min | one fit on 1.26 M rows |
| `--stage infer` (4 snapshots) | ~13 min | ~3 min per snapshot × 3,696 cells |

**Why so much faster than the CNN-LSTM branch's 10+ h:** this is a gradient-boosted tree ensemble on
hist binning, so a fit is ~50 s at any `n_estimators`, whereas a CNN-LSTM needs many epochs and each
epoch is minutes. The full ablation is the only long step, and it is a single command you can leave
running.

---

## What a run produces

**Everything comes from the command line, not the notebook.** The notebook is optional narrative; these
artifacts are written directly by the stages.

```
XGBoost/outputs/
├── ablation_summary.csv            THE HEADLINE TABLE: one row per arm, pooled RMSE / MAE / R2
├── ablation_results.csv            15 rows: per arm x per fold, tuned params, n_estimators
├── ablation_params.json            run configuration (feature lists, k, n_trials, partitions)
├── oof_predictions_{A,B,C,C+,TEXTURE}.csv   every scored row (~100 MB each)
├── test_metrics.csv                TEST-station read for --test-arm (RMSE/MAE/R2/IA/r/bias/within-5/10)
├── test_metrics.json               same, plus provenance
├── predictions_test.csv            per-row actual/predicted/residual on the reserved test stations
├── geocp_shared/
│   ├── geocp.json                  selected kernel+bandwidth, LOSO coverage, TEST coverage
│   ├── geocp_candidates.csv        every kernel x bandwidth candidate
│   └── predictions_test_intervals.csv
├── artifacts/latest/               model.joblib + run_info.json + features.json
├── surfaces/<date>/                per-cell CSV, GeoTIFF, AQI map, CONTINUOUS map, summary
│   ├── panel_<date>.png            4 snapshots, one shared colour scale
│   └── <date>_mean_continuous.png  24-hour mean
└── figures/
    ├── pred_vs_obs.png
    ├── residual_hist.png
    ├── accuracy_by_station.png
    ├── timeseries_sample.png
    ├── error_vs_concentration.png   <- where the RMSE comes from (the tail)
    ├── shap_importance.png          <- per-feature and per-family attribution (morphology share)
    └── geocp_coverage.png
```

This deliberately mirrors the CNN-LSTM run folder (`data/20261016_070432`: `metrics_summary.json`,
`tables/accuracy_summary.csv`, `plots/*.png`, `geocp.json`), so the two branches can be laid side by side.

**The numbers to quote:** `ablation_summary.csv` for the arm comparison, `test_metrics.csv` for the
headline accuracy on unseen stations, and `geocp_shared/geocp.json` for interval coverage.

## Run it from the notebook

`xgboost_pm25_pilot.ipynb` does the same work as the console and runs on **either** dataset. Open it and
Run All — a bootstrap cell sets the working directory, so it works from any launch directory.

| `XGB_DATASET` | What the notebook does |
|---|---|
| `full` (default) | trains all three arms on 83 cells, then GeoCP, SHAP and the surfaces panel |
| `pilot` | reproduces the published pilot narrative |

Two cells are written to stay tractable on the full network, and **both say so in their output**:

- **Cell 28** (bandwidth search) caps calibration scores at `MAX_CAL = 6000` with a seeded subsample.
  Uncapped it is O(n_cal × n_query) for every candidate `h` and every leave-one-cell-out fold; at 145 k
  calibration records it does not finish. This is the same cap the CLI applies via `--max-cal`.
- **Cell 29** (interval evaluation) caps test rows at `MAX_EVAL = 20000`, matching `--max-eval`.

If a cap engages, report it in the methods: the interval is still a valid split-conformal interval for
the data actually used, just estimated from fewer scores.

**The reuse guard is dataset-aware.** Cell 17 loads `outputs/` instead of retraining only when the
on-disk `oof_rows` equals *this* dataset's training-row count. So after a full-network run, opening the
notebook in pilot mode **retrains** rather than silently comparing 1.26 M full-network predictions
against 32 k pilot features. Set `REUSE_ANY_DATASET = True` in that cell to load them anyway.

Verified: on-disk pilot rows (32,083) vs full training rows (1,256,692) → the full notebook trains, the
pilot notebook reuses.

To exercise the notebook non-interactively (pilot only, ~10 min):

```
python test_notebook_flow.py
```


---

## Data quality: what the EDA changed

`eda/eda_results/EDA_Report.md` audits the raw network. Four recommendations, and what each is worth here:

| Finding | Rows | Applied | Effect |
|---|---|---|---|
| sensor-floor readings `<= 0.5 ug/m3` | 5,292 (0.36%) | **dropped before lagging** | stops a 0.0 becoming next hour's `pm25_lag_1`; model change ~0.3% |
| New Year's Eve fireworks window | 1,389 (0.09%) | **opt-in**, `XGB_CLEAN_NYE=on` | model unchanged (~0.3%); reported RMSE -1.9% and **R2 -0.010** purely from removing 363 extreme test rows |
| staggered QC cohorts (8 -> 40 sensors) | -- | lag requires >= 2 contributors | avoids deleting 2021-2023 |
| raw wide-format quirks | -- | not applicable | reads `QC Data Converted/` (long), not the wide files |

**The spatial lag now needs >= 2 reporting neighbours.** The cohort finding is not only about years: 18.4%
of cell-hours have exactly ONE contributing station, where a "neighbourhood" lag is really a copy of a
single neighbour. Those become NaN (`min_contributors=2` in `compute_spatial_lag`), which took C+'s lag
coverage from 89.9% to 81.6% of cell-hours and moved its test R2 from 0.707 to 0.672.

**NYE masking is off by default on purpose.** It buys RMSE by deleting the hardest rows, which is metric
cherry-picking unless stated; and 956.8 ug/m3 occurs outside that window, so the tail is not an artefact.
Turn it on for a sensitivity row: `set XGB_CLEAN_NYE=on`. Numbers from
`python experiments/eda_cleaning_impact.py --arm A --fold 0 --clean on|off`.

## Maps, and the optional basemap

`--stage infer` writes, per date: the per-cell CSV and GeoTIFF, an AQI-tier map, a **continuous-gradient**
map, a 2x2 **panel** of the four snapshot hours on one shared colour scale, and a **24-hour mean**.

The gradient is the point. AQI tiers (0-25 / 25-35 / ...) put nearly every cell of this pipeline's output in
the same "Good" bin, so the tiered map reads as one flat colour; the continuous scale is derived from the
data (p2-p98) and shared across the four panels so hours stay comparable. Sensor positions are overlaid as
white dots.

**The basemap needs no installation.** `maps/basemap.py` fetches OpenStreetMap tiles directly (standard
library `urllib`, stitched with matplotlib, cached in `.tile_cache/`) when `contextily` is absent, then
lightens them so the surface reads through at `alpha=0.78`. Tiles need internet at render time only; if the
fetch fails the maps render on plain axes instead of breaking.

`contextily` is still preferred when present. Set `basemap.configure(web_tiles=False)` for a fully offline
render.

## The 10 m surfaces, in the CNN-LSTM's cartography

The paper places the two branches side by side, so the XGBoost surfaces are drawn with the same furniture
as the CNN-LSTM figures (`model/output/cartography.py` in this repository): a light basemap, the NCR
boundary clip, city outlines and names, monitoring stations as white dots, kilometre axes, a two-tone
scale bar, a north arrow, one viridis colourbar and a "No data" swatch.

```bash
cd XGBoost

# 2x2 panel per date, scaled like the CNN-LSTM map of the same date
python maps/cnn_maps.py --days 20250830 --vmin 13 --vmax 47
python maps/cnn_maps.py --days 20250723 --vmin 10.4 --vmax 19.8

# one map per hour, at their zoom
python maps/cnn_maps.py --days 20250830 --single --vmin 13 --vmax 47
```

Two things have to exist before the first render:

| Needed | How to get it |
|---|---|
| `data/ncr_cities.gpkg` | `python -m preprocessing.boundaries` — fetches GADM 4.1, keeps the 17 NCR cities, 558 km². Not redistributed, so it stays out of git |
| basemap tiles | fetched once into `data/basemap_cache/` (Esri, no API key). Set `XGB_TILE_URL` to use another provider, e.g. CartoDB Positron with a free key |

### Filling the gaps to the boundary

By default the 10 m inference paints only pixels with complete morphology and a stand-in history — about
67% of the grid. `--fill` extends that to the whole study area, and the three steps are worth knowing
because they are a methodological choice, not a rendering trick:

1. **nearest valid morphology** — a pixel with no morphology (masked water, the DSM fringe) takes the
   features of its nearest valid pixel;
2. **edge-padded grid** — the stack is extended to the NCR bounding box (330 rows north, 548 south of the
   raster), so land beyond the raster still gets a prediction;
3. **in-paint and clamp** — any pixel still unpredicted takes the nearest predicted value, and every
   extrapolated pixel is limited to the 0.5–99.5 percentile range of the fully measured cells, so an edge
   artefact cannot invent a 290 µg/m³ cell.

```bash
python run_pipeline.py --stage infer10 --date 2025-08-30 --window-px 32 --fill
```

Coverage goes from 67% to 100% of the padded grid, with roughly 6.4 M of 13.0 M pixels extrapolated. Quote
that number wherever the surface is used: extrapolated cells belong in the mapped extent and **not** in any
accuracy statistic. Station-level metrics (test R², MAE, coverage, Moran's I) are unaffected by `--fill`,
because they come from `--stage test` on the 16 reserved stations.

## The shared-split rebuild (comparable to the CNN-LSTM run)

The CNN-LSTM branch ran on the same 83 usable stations and published a run in
`XGBoost/data/20261016_070432`. To compare architectures instead of splits, this branch can adopt their
**exact** station partition — 54 train / 13 calibration / 16 test — and score the same reserved test
stations they did.

```
cd XGBoost
$env:XGB_FOLDS = 'shared'                      # PowerShell;  export XGB_FOLDS=shared  on bash
python run_pipeline.py --stage all --no-infer
```

That single command runs: shared split -> 10 m patch statistics -> dataset report -> ablation ->
test-station evaluation -> GeoCP -> model artifact -> figures. `--no-infer` skips only the city-wide
surfaces (the slow tail); drop it to include them. Add `--smoke` to prove the chain in minutes,
`--n-trials 8` for a ~2 h read, or `--geocp-profile shared` to use their kernel x bandwidth search.

Its pieces are individually runnable:

| Stage | What it writes |
|---|---|
| `--stage split` | `data/fold_partitions_shared.json` (54/13/16, their station ids) |
| `--stage patch` | `data/patch_stats.csv` (35 features: 320 m @ 10 m, mean/std/min/max/p10/p50/p90 per band) |
| `--stage test` | `outputs/test_metrics.csv`, `predictions_test.csv` — the reserved test read |
| `--stage figures` | `outputs/figures/*.png` (6 figures) |
| `--stage geocp --geocp-profile shared` | `outputs/geocp_shared/` on their kernel/bandwidth grid |

**Five arms are available:** `A` (time only, 18), `B` (morphology + time, 15), `C` (full, 24),
**`C+`** (full + fold-aware spatial lag, 27) and **`TEXTURE`** (full + 10 m patch statistics, 59).

```powershell
$env:XGB_FOLDS='shared'
python run_pipeline.py --stage ablation --arms A B C C+ TEXTURE --n-trials 60 --row-stride 4
```

C+ computes its lag **inside each outer fold** from that fold's fit cells only (k=5 nearest stations,
matching the CNN-LSTM `spatial_lag: k=5`), so a held-out cell never contributes to its own feature. It
costs one `compute_spatial_lag` call per fold (~1-2 min), which is why it is noticeably slower than C.

In a 3-trial, single-fold smoke check on the shared split, C+ scored R² 0.479 against C's 0.513 — i.e.
the spatial lag did **not** help there. That is three tuning trials on one fold, so it is not a result;
it does mean the lag is no free win and the 20-trial run should decide it.

### What it already shows

On the **identical 16 test stations**, XGBoost arm C versus their CNN-LSTM:

| | XGBoost (C) | CNN-LSTM |
|---|---|---|
| RMSE | 6.560 | **6.132** |
| MAE | 3.516 | **3.512** |
| R² | 0.659 | **0.702** |
| Pearson r | 0.816 | 0.841 |
| bias | +0.716 | +0.128 |
| within 5 µg/m³ | 79.0% | 78.9% |
| within 10 µg/m³ | 94.5% | 94.6% |

**Typical error is a dead heat** (MAE 3.516 vs 3.512, within-5 and within-10 identical to 0.1 pp). The
CNN-LSTM's advantage is entirely in the tail: its RMSE and R² are better, and our bias is 5x larger,
because the tabular model under-predicts peaks. `outputs/figures/error_vs_concentration.png` shows it
directly — above 120 µg/m³ our RMSE reaches ~150 and bias ~+130.

That is the honest architecture comparison: **equal on typical hours, CNN better on extremes**, both
close to persistence.

### Verification

`python verify_shared_split.py` (24 checks) confirms the split is exactly 54/13/16, the test and
calibration stations are identical to theirs, no partition cell has NaN morphology, the folds are
leakage-free, and the texture table covers every in-split cell. Cells `80`, `169`, `219` are nodata in
the 10 m raster and had morphology IDW-imputed from neighbours — recorded in
`data/imputed_cells.json` and `data/imputed_texture_cells.json`.

## What changed, and why

### The new real-world data

| Source | Sensors | Rows | Coverage |
|---|---|---|---|
| `data/OpenAQ/` | 45 | 352,234 | 2024-08-13 … 2026-10-15 |
| `data/QC Data Converted/` | 40 | 1,011,214 | 2021-07-09 … 2025-12-31 |
| Combined | **86 locations** | 1.36 M | overlap: **506 days in both** |

Both arrive in one 8-column long schema (`location_name, location_id, sensor_id, utc_timestamp,
local_timestamp, pm25_value, parameter, unit`) — which nothing in this branch could read.
`ingest_sources.py` now converts them into the same wide schema as `Master_DL_Ready.csv`, so
`preprocessing/prep.py` and every later stage work unchanged.

**`location_id` is the key**, not `sensor_id`: it identifies *where* the station is, and where one
location reports under several sensor ids the readings are averaged within the hour.

**Short gaps are interpolated, long ones are not.** A gap of at most 24 h is linearly interpolated; a
longer outage stays entirely missing (deliberately *not* pandas' `limit=`, which would fill the first 24
hours of any gap and so fabricate data). This is the single biggest data-quality win: the median sensor
now keeps **99.8 % of its span**, against ~21 % under the pilot's no-interpolation rule, and the kept row
count rises from 43 k to 1.26 M.

### The morphology grid

`master_raster_v2.tif` (86.7 MB, 10 m, 5 bands, EPSG:32651, 27.3 × 39.0 km) supplies the morphology.
`build_morphology_grid.py` aggregates it into 500 m cells using the pilot's own rule — a fishnet over the
sensor bounding box, padded 2 km — and **the aggregator is validated against the published pilot grid**:

| Column | Pilot median | Rebuilt | Corr |
|---|---|---|---|
| `bh_mean` | 4.326 | 4.367 | **0.993** |
| `bh_stdev` | 4.861 | 4.429 | **0.997** |
| `ndvi_mean` | 0.194 | 0.193 | **0.982** |
| `bldg_dens` | 0.440 | 0.393 | 0.973 |
| `aspect_ratio_mean` | 0.310 | 0.901 | 0.616 ⚠ |

**Resolution stayed at 500 m deliberately.** Resolution changes what the features *mean*: at 500 m a cell
is a neighbourhood statistic, at 10 m it is closer to a single rooftop — and the existing 17 cells'
morphology is zonal statistics per 500 m cell. Going finer would break comparability, for a feature set
that carries ~2.5 % of model attribution.

> ⚠ **`aspect_ratio` uses a different convention and is the one unresolved item.** The pilot computed
> `mean(height over merged street buffers) / mean(road width)`; the raster's band is per-pixel
> `mean(H/W)` over road buffers. Measured over the pilot's 378 cells, the two differ by a median factor
> of **3.1**. Because `street_width_mean = bh_mean / aspect_ratio_mean`, the new run's street widths are
> internally consistent but **not on the pilot's scale**. Closing this needs the OSM road centrelines,
> which are not in the repository and could not be fetched (no network in this environment).

### The spatial split

`search_partitions` clusters for geographic *separation*, not size, so k = 3 on 83 cells gave
39 / 10 / 34 — asking for 59/12/12 produced train=10. `_balanced_partitions` now clusters finer and
groups whole clusters into the requested sizes, keeping each partition contiguous:

**64 train / 12 calibration / 7 test**, min separation 1,414 m, leakage test passes.

The pilot keeps the old path (`balanced=False`) so its published 11/3/3 is untouched.

---

## Results

### Pilot (the published record, unchanged)

| Arm | Features | Pooled RMSE | Pooled MAE | Pooled R² |
|---|---|---|---|---|
| **A TIME_ONLY** | 18 | 6.536 | 3.808 | **0.630** |
| **B MORPH_TIME** | 15 | 10.963 | 7.554 | **−0.040** |
| **C FULL** | 24 | 6.631 | 3.808 | **0.619** |

Morphology adds nothing measurable: arm B is worse than a constant, arm A matches arm C within noise,
and the six morphology features carry **2.5 %** of SHAP. The correct claim is *"no detectable
morphological signal at 11 training cells"* — a **power** statement, not a physical one.

### Full network — first read, smoke configuration

`--stage ablation --smoke` (1 trial, one outer fold, 452,060 scored rows, 64 training cells):

| Arm | RMSE | MAE | R² |
|---|---|---|---|
| A TIME_ONLY | 7.510 | 3.805 | 0.594 |
| **B MORPH_TIME** | **11.986** | 7.517 | **−0.033** |
| C FULL | 7.493 | 3.810 | 0.596 |

**The negative morphology result replicates at ~6× the network size.** Arm B is again below a constant,
and arm C again does not beat arm A. Treat these as a smoke-check, not a result: one tuning trial on one
fold is not the published protocol.

City-wide surfaces on 2026-04-11 08:00: **3,125 of 3,696 cells usable**, mean stand-in coverage **0.85**
(better than the pilot's 0.63), mean interval width 14.3 µg/m³.

GeoCP selected **h = 750 m** and again flagged **DEGRADED** — no bandwidth reached 90 % on every
calibration fold, so the unweighted split-conformal interval remains the primary calibrated result. The
pilot's structural finding holds at this scale too.

---

### Five arms on the shared split — the current result

20 trials per arm, 3 outer folds, 54 training stations / 881,981 scored rows, on the split shared with
the CNN-LSTM run (`data/20261016_070432`).

| Arm | Features | Pooled RMSE | Pooled MAE | Pooled R² | mean R² (sd) |
|---|---|---|---|---|---|
| **A TIME_ONLY** | 18 | **9.329** | **3.953** | **0.569** | **0.611 (0.105)** |
| **B MORPH_TIME** | 15 | 14.607 | 8.623 | **−0.057** | −0.040 (0.063) |
| **C FULL** | 24 | 9.380 | 3.962 | 0.564 | 0.606 (0.110) |
| **`C+` + spatial lag** | 27 | 9.743 | 4.025 | 0.530 | 0.574 (0.117) |
| **TEXTURE** | 59 | 9.532 | 3.988 | 0.550 | 0.592 (0.116) |

**Arm A wins, and every added feature set makes it slightly worse.**

1. **Morphology alone cannot predict — third replication.** B is below a constant at **−0.057** pooled
   (−0.040 mean), matching the pilot (−0.040) and the 64-cell run (−0.039). In one fold its
   `n_estimators` came out at **15**: the model barely builds a tree.
2. **The spatial lag hurts** (0.530 vs C's 0.564). Neighbourhood readings are a smoothed copy of what a
   station's own lags already hold, and they add noise.
3. **10 m patch texture hurts slightly** (0.550 vs 0.564) even with 59 features and access to
   within-cell spread and extremes. This is the fair architecture test: XGBoost was given the same
   local information the CNN convolves, and it did not convert it into skill.
4. **The folds differ enormously** — per-fold R² for arm A is 0.516 / 0.724 / 0.591. Report
   **mean ± sd (0.611 ± 0.105)**, not the pooled 0.569 alone, or the two will look inconsistent.

Against the CNN-LSTM on the same 54 stations: their OOF R² is **0.633** vs our **0.611**, on
819,388 vs 881,981 rows (same stations, each branch's own row filtering). Their test R² on the 16 shared
stations is **0.702**; ours, measured with the previous ablation's settings, was **0.659** — re-run
`--stage test` to get the figure for the winning arm.

## How the pieces fit

- **`data.py`** resolves every path and selects the dataset. `XGB_DATASET=full|pilot`.
- **`ingest_sources.py`** — raw per-sensor CSVs → wide table. Flags: `--sources OpenAQ QC`, `--max-gap N`,
  `--out PATH`.
- **`build_morphology_grid.py`** — raster → grid. Reuses `preprocessing/grid_builder.py`
  (`build_vector_grid`, `aggregate_morphology`, `coverage_report`), which also accepts an explicit
  `anchor` + `nrows`/`ncols` so a known grid can be reproduced exactly (that is how the pilot was
  validated).
- **`preprocessing/prep.py`** — runs on import, defines `df_clean`, `X`, `y`, `features`, `groups`.
  Recomputes lags/rolling stats per sensor on a gap-aware hourly index (replacing the source file's naive
  `pm25_t-1/2/3` row shifts) and drops rows whose window crossed a gap.
- **`training/ablation.py`** — arms A/B/C, nested spatial CV, Bayesian search with Expected Improvement.
- **`uncertainty/geocp.py`** — GeoCP with the thesis's equations; `--max-cal` / `--max-eval` exist because
  the weighted quantile is O(n_query × n_cal), which is intractable at 145 k calibration records.
- **`training/artifacts.py`** — saves `model.joblib` + `run_info.json`; refuses to load a model whose
  feature list does not match the caller's.
- **`inference/cell_inference.py`** + **`maps/surfaces.py`** — every cell gets a prediction, a GeoCP
  interval and a stand-in coverage diagnostic; written as per-cell CSV, GeoTIFF, AQI map, continuous map.

## Known limitations

1. **`aspect_ratio` convention gap** (3.1×) — see above. Needs OSM streets.
2. **City-wide maps are stand-in driven and read low.** At a cell with no sensor the nine history
   features come from an inverse-distance mean of the nearest training stations; PM2.5 is right-skewed,
   so averaging removes peaks. On the pilot this read 9.75 µg/m³ against 15.28 from real histories.
   Morphology's ~2.5 % share means these are closer to a smoothed interpolation than a morphology model.
3. **Full ablation takes ~2.5–4 h** at 1.25 M rows (measured, see timings).
4. **GeoCP subsamples** at this size (`--max-cal`), which is reported in `geocp_config.json`.
5. **Test partition is 7 cells** — the balanced grouping overshot train to 64, and 7 is enough for a
   final read but leaves limited power. Re-run `--stage partitions --n-calibration 10 --n-test 12` to
   rebalance if preferred.
6. **Three locations are excluded, all explained:** `4797656` (Muntinlupa, 2.7 km south of the raster),
   `9000015` and `9000025` (no building data in the raster).
7. **The two sources are not harmonised.** Report per source where it matters; the 506 shared days allow
   a like-for-like comparison.

## Verification

```
python run_pipeline.py --stage checks      # all seven, pass/fail summary
```

| Check | What it protects |
|---|---|
| `checks/test_spatial_lag.py` | spatial-lag regression test (caught the `weights` bug that made every lag 1.0) |
| `checks/verify_ablation.py` | no held-out cell in training; arms on identical rows; metrics reproduce |
| `checks/adversarial_step3.py` | tries to falsify the morphology result three ways |
| `checks/check_output_coherence.py` | output sanity + morphology-swap perturbation |
| `checks/check_morphology_coverage.py` | does the sensor network span the raster's morphology |
| `checks/verify_geocp.py` | 28 checks incl. the flat-kernel limit |
| `checks/test_notebook_flow.py` | runs the notebook's cells in order |

**Known gap:** the checks verify cell reuse but nothing verifies the grid is *geographically* correct
(CRS, north-up orientation, completeness). That was established by one-off analysis — CRS confirmed as
EPSG:32651 by reverse-engineering it from the sensors' own lat/lon vs x/y, orientation confirmed north-up
(`corr(row_index, northing) = −1.000`) — and a `checks/verify_georeference.py` would lock it down.
