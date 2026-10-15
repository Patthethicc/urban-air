# XGBoost branch — street-level PM2.5 from urban morphology

Street-level PM2.5 estimation from urban morphology and temporal features, using XGBoost with nested
spatial cross-validation, Bayesian hyperparameter optimisation, a three-way ablation, geographically
weighted conformal prediction intervals, and city-wide inference.

18 sensors · 17 grid cells · 43,273 hourly observations after cleaning · 685 days
(2024-08-16 to 2026-07-03).

> **Status.** Steps 1–5 and 7–8 of the work plan (`00_XGBOOST_PLAN.pdf`) are complete: data preparation,
> spatial partitions, the ablation, GeoCP, the Deliverable 1 dataset report, the saved model artifact,
> and city-wide inference with per-cell intervals and AQI maps. **Step 6 (ground validation) is blocked**
> — it needs deployed-sensor data and the written confirmation of the revised Deliverable 6. The
> manuscript text itself is not written (the remainder of Step 7).
>
> **Everything below is on held-out grid cells**, never on held-out rows. `STATUS.md` carries the running
> state and the cumulative list of decisions that affect the manuscript; `STEP3_ABLATION.md` and
> `STEP4_GEOCP.md` are the methods records for their steps. This README is both the write-up and the map
> of the repository.

---

## 1. The headline result, and why it is not the obvious one

The ablation asks three questions on identical folds, with the calibration and test cells kept out of
cross-validation, tuning and fitting entirely:

| Arm | Features | Pooled RMSE | Pooled MAE | Pooled R² |
|---|---|---|---|---|
| **A TIME_ONLY** — cyclical time + PM2.5 history | 18 | 6.536 | 3.808 | **0.630** |
| **B MORPH_TIME** — morphology + cyclical time, no PM2.5 history | 15 | 10.963 | 7.554 | **−0.040** |
| **C FULL** — everything | 24 | 6.631 | 3.808 | **0.619** |

(reference: predicting a constant scores R² = 0.000)

**The model's skill is temporal persistence, not urban form.**

- **Arm B scores below a constant.** Morphology plus time of day, with no PM2.5 history, is *worse* than
  predicting the mean, and worse than the training-fold mean on two of three folds.
- **Arm A matches arm C.** The full model gains nothing detectable from its six morphology features and
  three morphology × hour interactions. A's pooled advantage over C is +0.011 R², which is **smaller
  than the per-fold spread of either arm** (sd 0.035–0.040 across three folds) — so the correct
  statement is that morphology adds nothing measurable, not that it actively hurts.
- SHAP agrees by an independent route: the six morphology features together account for **2.5%** of
  total mean |SHAP|, and `pm25_lag_1` alone outweighs all of them.

**What this does and does not mean.** Three independent checks were run, and the result survived all of
them: a ridge regression on the same features reaches R² 0.07 *in-sample* with morphology (so this is
not a model-choice problem), and a 1-nearest-neighbour lookup in standardised morphology space
correlates **−0.003** with cell-level mean PM2.5 — morphology does not even *rank* the cells correctly.

But the honest reading is a **power** statement, not a physical one. The 17 sensors span 68–168% of the
raster's per-cell morphological variation, so coverage is not the binding constraint; the sample size is
(11 training cells, 3 test cells). With that many cells, "no effect" and "effect too small to detect"
are indistinguishable. The right claim is *"no detectable morphological signal at 11 training cells"*,
**not** *"urban form does not affect street-level PM2.5"*.

**Consequence for the thesis:** the benchmark the CNN-LSTM branch must beat is **arm A**, not arm C. A
sequence model handed lagged PM2.5 and compared against arm C would look good for the wrong reason.

> **Reproducibility note, measured rather than assumed.** Re-running `ablation.py` reproduces arms B and
> C **exactly** but moves arm A (pooled R² 0.638 → 0.630 between two runs): its Bayesian search selects
> different hyperparameters despite the fixed `random_state`, because parallel XGBoost fits update the
> scikit-optimize surrogate in a non-deterministic order. The conclusion is unaffected (B is below a
> constant, and A matches C, in both runs), and `geocp.py` depends only on arm C, whose parameters are
> stable. Quote the table above as one run of a search, not as a bit-reproducible figure.

## 2. Prediction intervals (GeoCP)

Nominal coverage 90% (α = 0.10), calibrated on 3 cells the model never trained on, evaluated on 3
further cells it never saw:

| Method | Empirical coverage | Mean width | Median width | Width / prediction |
|---|---|---|---|---|
| **Unweighted split conformal** | **0.862** | 19.375 | 19.375 | 1.873 |
| GeoCP, thesis-specified bandwidth search (h = 500 m) | **0.693** | 11.531 | 10.380 | 1.095 |
| GeoCP, geometry-only bandwidth (h = 4,528 m) | 0.842 | 18.003 | 18.058 | 1.739 |

**The thesis-specified bandwidth rule does not work at this network density, and the pipeline says so
itself.** Given a grid of candidate bandwidths including infinite (i.e. plain split conformal), the
selection picks the narrowest one — h = 500 m — which concentrates all weight on a single calibration
cell (effective number of calibration cells = **1.00 of 3**, all three test cells dominated by cell
103, whose mean error is the lowest of the three at 3.04 µg/m³). The half-width collapses to 5.19–6.87
µg/m³ while the test cells' actual mean absolute error is 4.87–5.29. The 40% narrower intervals are
**under-coverage, not efficiency**: coverage falls to 0.693, and 111 test rows that split conformal
covers are missed, with none gained.

No candidate reaches 90% on every hold-out calibration cell, so the run is flagged `DEGRADED` and
reports the unweighted split-conformal interval as the primary calibrated result, with GeoCP as a
pilot. That is the finding, not a bug to be tuned away: **three calibration sites 3.8–8 km apart cannot
support a geographically weighted quantile.** At the geometry-only bandwidth the weighting genuinely
pools the sites (2.63 of 3 effective cells) and buys 7.1% narrower intervals for 2.0 points of coverage.

Every configuration, including plain split conformal, **under-covers** the 90% target. With 3
calibration and 3 test cells differing 2× in error level, exchangeability across cells is weak. Report
the shortfall; do not quote the nominal 90%.

Lower bounds are clipped at zero **for display only**; coverage is always computed on the unclipped
interval (thesis Section 3.5.2). 471 of 5,420 lower bounds fall below zero.

## 2b. City-wide surfaces, and why they read low

`run_pipeline.py --stage infer --date YYYY-MM-DD` predicts **every cell of the morphology grid**
(378 for the pilot) at the four thesis snapshot hours (08:00 / 12:00 / 17:00 / 21:00), with a GeoCP
interval per cell, writing a per-cell table, an AQI map and a summary per snapshot.

The model's nine PM2.5-history features do not exist at a cell with no sensor, so they are filled with
an inverse-distance **stand-in** from the k = 4 nearest proper-training stations — the approach the
CNN-LSTM branch uses. Morphology, cyclical time and the interaction columns are exact.

**That stand-in biases the surface low, and it is measurable.** For 2025-07-23 08:00:

| | mean | median | max |
|---|---|---|---|
| Observed (sensor rows) | 16.11 | 14.50 | 333.00 |
| Same model on **real** sensor history | 15.28 | 14.47 | 89.61 |
| **Stand-in surface** (378 cells) | **9.75** | 9.28 | **21.80** |

The stand-in is an inverse-distance *mean* and PM2.5 is strongly right-skewed, so averaging removes the
peaks a map exists to show. The surface also carries `standin_coverage` — the fraction of the required
73-hour history that neighbours actually reported, mean **0.63**, not 1.0. Between that and morphology's
~2.5% share of model attribution, **these maps are closer to a smoothed interpolation than to a
morphology prediction**, and any published map must say so.

## 3. Spatial bias

Global Moran's I on per-cell mean out-of-fold residuals over the 11 training cells:
**I = −0.229, p = 0.273** → no statistically significant spatial clustering. Per-cell mean residuals
are small and all positive (+0.30 to +1.60 µg/m³), so the model neither systematically fails in one
part of the city nor carries a large location-specific bias. Residuals are `actual − predicted`.

## 4. Data

| | |
|---|---|
| Raw hourly readings | 122,841 |
| **Kept** after gap-aware lag cleaning | **43,273** (35.2%) |
| Dropped | 79,568 |
| Sensors | 18, all inside the raster |
| Grid cells with a sensor | 17 of 378 raster cells |
| Median per-sensor completeness, raw | 70.4% of its span |
| Median per-sensor completeness, kept | 21.3% of its span |

Full details in `XGBoost/outputs/dataset_report.md` (Deliverable 1).

**Why 65% of rows are dropped — and why that is deliberate.** `Master_DL_Ready.csv` carries its own
`pm25_t-1/2/3` columns, which are naive row-shifts on irregularly timed data: during a gap they pair a
reading with one from up to 29 hours earlier, silently mislabelling a stale value as a 1-hour lag. This
pipeline recomputes every lag and rolling statistic on a per-sensor hourly index instead, so a window
that crosses a gap becomes missing and the row is dropped. The drop funnel, per requirement:

| Requirement | Rows kept | Lost |
|---|---|---|
| With a target value | 122,841 | — |
| + 1 h lag | 100,711 | 22,130 |
| + 2 h lag | 83,887 | 16,824 |
| + 3 h lag | 71,046 | 12,841 |
| + 4 h lag | 61,106 | 9,940 |
| + 24 h rolling mean | 51,205 | 9,901 |
| + 72 h rolling mean | **43,273** | 7,932 |

Sensors report ~70% of the hours in their span, so requiring *consecutive* hours compounds quickly —
only ~21% of a sensor's span survives. Whether to relax the 72 h window is a modelling decision taken
with open eyes; the alternative keeps far more rows but feeds the model mislabelled lags.

PM2.5 on kept rows: mean 16.47, median 14.70, sd 11.37, p90 29.5, max 333.0 µg/m³.

**Grid cell 78 holds two sensors** (3370151, 6303934) reporting at the same hours. Its 5,501 rows are
12.7% of the kept data, and 105 cell-hours are duplicated, so every pooled metric weights that cell
roughly twice. No aggregation is applied — collapsing it would change every number reported here. The
spatial-lag feature does collapse the pair by mean, so a two-sensor cell cannot double-count inside it.

## 5. Method

- **Target:** PM2.5 in µg/m³, trained as `log1p(PM2.5)` and inverse-transformed with `expm1` before
  every reported metric.
- **Spatial partitions** (`data/fold_partitions.json`): 11 training / 3 calibration / 3 test cells,
  chosen by K-Means over cell centroids as the best of 200 seeds by partition separation. The minimum
  distance between any two partitions is 1,500 m. Partitions are assigned by cell, so all rows of a cell
  stay together. No data from a held-out cell enters cross-validation, tuning or fitting.
- **Nested spatial CV:** outer k = 3, inner k = 3, with the inner folds recomputed inside each outer
  fold so the outer test cells never influence the search. k = 3 rather than 4 because 11 training cells
  would otherwise leave 2–3 cells per outer fold.
- **Tuning:** Bayesian optimisation with **Expected Improvement** (`acq_func='EI'`, thesis Eq. 3.28),
  8 random starts, 20 calls per arm per outer fold, `n_estimators` from early stopping (cap 500,
  patience 20) averaged over inner folds. Tuned **per arm**, so no arm inherits another arm's optimum.
- **Verification:** each component has a dedicated check; see section 9.

## 6. Known limitations

1. **Pilot scale.** 11 training cells, 3 calibration cells, 3 test cells. Every claim here is a pilot
   claim. The morphology question needs more sensors *inside* a consistently built raster.
2. **Persistence, not morphology.** The full model's accuracy comes from PM2.5's own recent history
   (corr(`pm25_lag_1`, PM2.5) = 0.841). Stated plainly rather than buried in a high R².
3. **"New cell, known sensor history."** The lag features are each sensor's *own* history, so a
   held-out cell is still scored using its own past readings. This measures generalisation to a new
   location *with an instrument*, not to an unmonitored one. The spatial lag (arm C+) is what would
   address it.
4. **GeoCP under-covers at the specified bandwidth** (0.693 vs 0.90), and no configuration reaches the
   target. Report the shortfall.
5. **Thin inner folds.** Two outer folds contain an inner fold holding a single cell, so inner-CV model
   selection is noisy.
6. **Cell 78 double-weighted** (12.7% of rows; see section 4).
7. **Arms differ in tuning as well as features**, since each arm was tuned separately. A
   shared-parameter run would be the tighter control if a reviewer asks.
8. **City-wide maps are stand-in driven and read low** — mean 9.75 vs 15.28 µg/m³ from real histories,
   with the range crushed from 4–333 to 4–22. Neighbour coverage is 0.63, not 1.0. See section 2b.
9. **The Bayesian search is not bit-reproducible for arm A** (pooled R² 0.638 → 0.630 across two runs).
   Arms B and C reproduce exactly, and `geocp.py` uses arm C, so the Step 4 numbers are stable.

## 7. Repository layout

Data lives in `data/`; code is grouped by what it does, mirroring the CNN-LSTM branch. Root-level
scripts are thin shims so every command named in the plan still works.

```
thesis_model/
│
├── 00_XGBOOST_PLAN.pdf              the work plan (steps 1-8)
├── README.md                        this file — write-up and repository map
├── STATUS.md                        state, decisions, cumulative manuscript-affecting changes
├── STEP3_ABLATION.md                step 3 methods + results
├── STEP4_GEOCP.md                   step 4 methods + results
├── CNN-LSTM/                        the other branch (own README)
│
└── XGBoost/
    ├── run_pipeline.py              staged entry point (see section 9)
    ├── data.py                      resolves every input/output path, so scripts run from anywhere
    │
    ├── data/                        ALL inputs
    │   ├── Master_DL_Ready.csv            18.0 MB  122,841 hourly readings, 18 sensors
    │   ├── full_morphology_grid.csv       51 KB    378 raster cells @ 500 m
    │   ├── fold_partitions.json           1.9 KB   the frozen 11 / 3 / 3 split
    │   └── sensor_to_grid_map_full.csv    3.4 KB   derived, rewritten on every import
    │
    ├── preprocessing/
    │   ├── prep.py                  data prep, features, gap-aware lags, spatial lag
    │   ├── spatial_cv.py            cell partitions, fold maps, JSON persistence
    │   └── spatial_bias.py          Moran's I / LISA on residuals
    │
    ├── training/
    │   ├── ablation.py              feature arms + nested spatial CV
    │   ├── partitions.py            rebuilds and validates the split
    │   ├── artifacts.py             step 7: model.joblib + run_info.json + features.json
    │   └── ablation_outputs.py      guard for the regenerable large outputs
    │
    ├── uncertainty/
    │   └── geocp.py                 GeoConformal Prediction + quantile_grid
    │
    ├── inference/
    │   └── cell_inference.py        step 8: all 378 cells, stand-in history
    │
    ├── maps/                        (named maps/, not output/: results live in outputs/)
    │   └── surfaces.py              step 8: per-cell tables, AQI maps, summaries, GeoTIFFs
    │
    ├── reporting/
    │   └── dataset_report.py        Deliverable 1
    │
    ├── checks/                      the seven verification scripts (section 9)
    │
    ├── configs/                     per-stage settings
    ├── xgboost_pm25_pilot.ipynb     the narrative and the figures
    │
    ├── outputs/
    │   ├── README.md                what is committed vs regenerated
    │   ├── ablation_{results,summary}.csv   the ablation numbers
    │   ├── dataset_report.md        Deliverable 1
    │   ├── artifacts/<run_id>/      model.joblib + run_info.json + features.json
    │   ├── surfaces/<date>/         per-snapshot CSV, AQI PNG, summary JSON
    │   └── geocp/, geocp_median_h/  step 4 intervals and coverage
    │
    └── <root shims>                 prep_n_proc_pilot.py · geocp.py · ablation.py
                                     + one per check script
```

**Not listed because they are not needed to run or read the pipeline:**
`full_morphology_grid_wkt.csv` (no code reads it; GIS reference), `.ipynb_checkpoints/` (Jupyter
autosave), `__pycache__/`, and — in the wider repository — `CNN-LSTM/pilot testing/` (raw source data
already represented inside each branch).

### Inputs

| File | Rows | What it is |
|---|---|---|
| `data/Master_DL_Ready.csv` | 122,841 | Hourly PM2.5 per sensor: `location_id`, `local_timestamp`, `pm25_target`, coordinates, and the CNN-LSTM branch's own `pm25_t-1/2/3` columns (**replaced** by this pipeline, see section 4). |
| `data/full_morphology_grid.csv` | 378 cells | Morphology per raster cell: `bh_mean`, `bh_stdev`, `ndvi_mean`, `bldg_dens`, `aspect_ratio_mean`, plus cell bounding boxes and `row_index`/`col_index`. 17 cells contain a sensor. |
| `data/fold_partitions.json` | 11 / 3 / 3 | The frozen cell split. Every later step reads it; regenerating it re-clusters the experiment. |

### Modules

**`preprocessing/prep.py`** — runs **on import** and defines `df_clean`, `X`, `y` (log1p), `features`,
`groups`, which is why every other script just imports it. It spatial-joins each sensor to its raster
cell (rewriting `sensor_to_grid_map_full.csv`), derives `street_width_mean`, builds cyclical time and
the three morphology × hour interactions, recomputes lags and rolling stats per sensor on a gap-exposed
hourly index, log1p-transforms the target and drops rows whose window crossed a gap → `X = (43273, 24)`.
It also defines `compute_spatial_lag(...)`, the fold-aware inverse-distance feature (thesis Section
3.1.5), which is deliberately **not** in `features` because it is fold-dependent — it is arm C+.

**`preprocessing/spatial_cv.py`** — `get_grid_centroids()`, `assign_spatial_folds()`,
`get_outer_folds(exclude_grid_ids=...)`, `get_inner_folds()`, `get_partitions()`/`search_partitions()`
(the 11/3/3 split), `build_fold_document()`/`save_fold_map()`/`load_fold_map()` (JSON persistence;
`load_fold_map` **refuses** a document with no `partitions` key so a later step cannot silently treat
every cell as training data), and `feasibility_report()`.

**`preprocessing/spatial_bias.py`** — `aggregate_residuals_by_grid()`, `build_knn_weights()`,
`global_morans_i()`, `local_morans_i()`, in numpy so libpysal/esda are not required. Called from the
notebook.

**`training/ablation.py`** — defines `ARM_FEATURES` = {A: TIME_ONLY 18, B: MORPH_TIME 15, C: FULL 24,
C+: 27} and `validate_arm_features()`, which asserts the groups cover the import-time feature list
exactly. `tune_and_evaluate()` is the nested CV core (per outer fold: EI search on inner folds, then fit
and score the held-out fold in raw µg/m³ after `expm1`); `run()` loops the arms, checkpoints after every
fold, and asserts all arms were scored on identical rows.

**`training/partitions.py`** — rebuilds `fold_partitions.json` and validates it: spatial-lag coverage per
partition, the leakage test (corrupt each held-out cell's own readings and confirm its lag is unchanged),
contributor counts, determinism, JSON round-trip.

**`training/artifacts.py`** — step 7. `save_model_artifact()` writes `model.joblib` + `run_info.json` +
`features.json`. `run_info.json` carries the feature order (positional — wrong order gives
plausible-looking wrong numbers), hyperparameters as fitted, `n_estimators`, the arm and outer fold they
came from, the target transform, the residual convention, the partitions and fold maps, and library
versions. `load_model_artifact(..., expect_features=...)` **refuses** a feature-list mismatch, because
that failure is silent rather than loud.

**`uncertainty/geocp.py`** — implemented from the thesis, not approximated: nonconformity scores
(Eq. 3.36), metre distances (Eq. 3.40), kernel weights (Eq. 3.42) normalised to unity (Eq. 3.41/3.43),
weighted quantile (Eq. 3.44/3.45), symmetric intervals (Eq. 3.46–3.50), 90% default target (Section
4.5.4), coverage on the **unclipped** interval (Section 3.5.2), bandwidth by grid search on held-out
calibration cells (Section 4.5.3), and Table 4.2 metrics. `quantile_grid()` produces exact half-widths
for many query points at once, which is what makes the step 8 surfaces possible.

**`inference/cell_inference.py`** — step 8. Per-cell predictions for the whole raster, filling the nine
history features from the stand-in (section 2b).

**`maps/surfaces.py`** — step 8 presentation: per-cell CSV (always), GeoTIFF (when `rasterio` is
installed), AQI map PNG, and a summary JSON carrying the section 2b caveat.

> **Why `maps/` and not `output/`.** The generated results live in `outputs/`, so a code package called
> `output/` sat one character away from a data folder called `outputs/` — easy to confuse, and neither
> name said which was which. The CNN-LSTM branch calls its equivalent package `output/`; that branch is
> untouched, because renaming tested code to fix a naming collision on our side is a bad trade. The
> folders therefore occupy the same position in both trees but differ in name, deliberately.

**`reporting/dataset_report.py`** — Deliverable 1. Re-runs the prep stages and reports rows kept/dropped
with a per-requirement drop funnel, sensors, cells, date range, missingness and feature distributions.

## 8. The notebook

`XGBoost/xgboost_pm25_pilot.ipynb` (35 cells) is the narrative and the figures, **not** the source of
the numbers. Section headings: 1 Setup, **2 Data prep**, **2b Dataset report**, 3 Fold assignment,
**3b Partitions + spatial lag**, **3c Feature arms**, **4 Nested CV / ablation**, **5 Ablation results**,
6 Spatial bias, **7 Production model**, **8 GeoConformal Prediction**, 9 SHAP, 10 Summary.

A bootstrap cell (cell 2) sets the working directory and `sys.path`, so the notebook runs from any
launch directory. Section 8 calls the tested `geocp.py` functions rather than re-implementing the
method, so the notebook and the CLI cannot drift apart.

The authoritative numbers are the CSVs in `outputs/`. The notebook's saved outputs predate Steps 3–4,
so **it needs one top-to-bottom run** (about 7 minutes from a clean clone, most of it the ablation cell)
before its figures reflect the current pipeline. That is why the checks, not the notebook, are the
verification path.

## 9. Verification and how to run

```
cd XGBoost
pip install xgboost scikit-optimize scikit-learn pandas numpy matplotlib shap
pip install rasterio affine          # optional: only for GeoTIFF output in the infer stage
```

### Run the pipeline

```
python run_pipeline.py --stage report       # Deliverable 1 dataset report
python run_pipeline.py --stage partitions   # validate the 11 / 3 / 3 split
python run_pipeline.py --stage ablation     # ~7 min: the three-way ablation
python run_pipeline.py --stage geocp        # conformal intervals and coverage
python run_pipeline.py --stage save         # step 7: model artifact
python run_pipeline.py --stage infer --date 2025-07-23   # step 8: city-wide surfaces
python run_pipeline.py --stage checks       # every verification script
```

`--stage all` runs report → partitions → ablation → geocp → save. **Order matters:** `ablation` must
precede `geocp` (which loads tuned parameters from `outputs/ablation_results.csv`) and `save`/`infer`
(which take the production model's configuration from the same file).

Every stage also runs as a script under its plan filename — `python geocp.py --alpha 0.10`,
`python ablation.py --arms A B C` — through the root shims.

### The checks

| Check | What it protects |
|---|---|
| `checks/test_spatial_lag.py` | Regression test for `compute_spatial_lag`: varying output, exact row counts, agreement with a hand-computed weighted mean, no self-inclusion, and that the feature is neither constant nor a copy of the target. **This is the test that caught the `weights`/`DataFrame.weights` bug that silently made every lag value 1.0.** Run after any edit to that function. |
| `checks/verify_ablation.py` | No calibration/test cell entered training; all arms scored on identical rows; stored metrics reproduce from stored predictions; residual sign; per-fold spread reported. |
| `checks/adversarial_step3.py` | Tries to **falsify** the negative morphology result with a linear model, a 1-NN morphology lookup, a reachability check, and per-fold noise bounds. Run before quoting that result. |
| `checks/check_output_coherence.py` | Outputs mechanically sane: bounded predictions, whether arm C actually reads its morphology columns (prediction agreement + a morphology-swap perturbation), the in-sample/out-of-fold gap. |
| `checks/check_morphology_coverage.py` | Whether the 17 sensors span the raster's morphological variety — what lets the negative result be worded as a power statement rather than a physical claim. |
| `checks/verify_geocp.py` | 28 checks: the flat-kernel limit (the uniform kernel must reproduce split conformal exactly), the weighted quantile against a brute-force implementation, partition discipline, interval arithmetic and the clipping policy, monotonicity in α, determinism. |
| `checks/test_notebook_flow.py` | Reproduces the notebook's state faithfully and executes its cells in order. Caught the missing `cx`/`cy` columns that broke the GeoCP cells on the first real run. |

Three of them read the per-row predictions, which are **not committed** (8.4 MB, regenerable). They stop
with one actionable message telling you to run `--stage ablation` first. See `XGBoost/outputs/README.md`.

### Generated and derived files

| File | Produced by | Notes |
|---|---|---|
| `data/sensor_to_grid_map_full.csv` | `preprocessing/prep.py` on import | Safe to delete; rewritten every run. |
| `data/fold_partitions.json` | `training/partitions.py` | **Authoritative split.** Regenerating it changes the experiment. |
| `outputs/ablation_*.csv`, `oof_predictions_*.csv` | `training/ablation.py` | The ablation numbers, and the per-row predictions the checks use. |
| `outputs/geocp*/*` | `uncertainty/geocp.py` | Intervals, coverage, bandwidth curve, config. |
| `outputs/artifacts/<run_id>/` | `run_pipeline.py --stage save` | Model + run metadata. |
| `outputs/surfaces/<date>/` | `run_pipeline.py --stage infer` | Per-cell tables, AQI maps, summaries. |

## 10. Growing beyond the current grid

The 378-cell raster covers 10.5 × 9 km at 500 m. A sensor whose coordinates fall outside that bbox is
**dropped, in two steps that only announce themselves once**: `prep.build_sensor_to_grid_map` finds no
containing cell and prints `WARNING: sensor ... falls outside full_morphology_grid.csv -- dropped`, then
the `how='left'` morphology merge leaves its six morphology columns NaN and `dropna` removes the rows
silently. All **18 current sensors are inside** the grid, so nothing is being lost today.

This is a **prep** boundary, not a model one: the morphology features are generic cell attributes and no
column of the 24-feature matrix is grid-unique, so any sensor assigned to a cell trains normally.
Training on a sensor outside therefore needs only that the grid be extended to contain it (below) and
the partitions be re-searched, since the cell count changes. The one real constraint is that every
sensor and cell share a **single projected CRS**, because `geocp.py` computes metre distances
(Eq. 3.40) and `spatial_cv.py` clusters centroids in metres.

There is no larger morphology file in the repository. Rather than search for one, **build it**: the
CNN-LSTM branch's `preprocessing/master_raster_builder.py` + `raster_harmonizer.py` produce a
multi-band GeoTIFF to any extent and resolution from source vectors, driven by these `data.yaml` keys —
all currently `null`:

| Key | Source it expects |
|---|---|
| `buildings_csv` | Google Open Buildings, WKT geometry |
| `streets` | OSM road centerlines (gpkg / shp / geojson) |
| `height_raster` | 4 m building-height raster |
| `ndvi_raster` | 10 m Sentinel-2 NDVI |
| `boundary` | NCR study-boundary polygon |

Its output bands are `height, ndvi, building_density, street_width, aspect_ratio, height_std`, and
`height_std_neighborhood_m: 500` is documented there as *"the footprint of the XGBoost side's
`bh_stdev`"* — so morphology built for new cells stays comparable with the existing 17. Filling those
five keys and running that branch's `--stage rasters` is a **GIS data-acquisition task, not a code
change**; afterwards, add the new sensors to `data/Master_DL_Ready.csv`, point `data.py` at the new
grid, and re-run the stages.
