# Reading guide and glossary

Companion to `xgboost_full.ipynb`. The notebook is the thing that runs; this file explains what each cell
does and what the numbers in its output mean.

---

## 1. What each cell does

Run the notebook top to bottom. Cells that depend on an earlier result say so.

| # | Section | The cell does | Main output |
|---|---|---|---|
| 0 | Environment | finds the code, finds the data, builds derived tables if absent, installs `scikit-optimize`/`shap` | prints `root`, `folds`, `master`, `grid` |
| 1 | Data and features | imports `preprocessing.prep`, which merges the 500 m morphology grid onto the sensor table, engineers gap-aware lags and cyclical time terms | `df_clean` — one row per station-hour with 24 features |
| 1b | EDA plots | PM2.5 distribution, rows per grid cell, rows by network | three plots |
| 1c | Cleaning policy | restates what the EDA cleaning removed and how much error sits in the tail | row counts, tail share |
| 2 | Dataset report | Deliverable 1: coverage, station counts, dropped sensors | `outputs/dataset_report.md` |
| 3 | Spatial split | writes the 54/13/16 station split and the 10 m patch statistics, then draws the fold map | `data/fold_partitions_shared.json`, `data/patch_stats.csv` |
| 4 | Ablation | five feature arms x 3 outer folds x 20 Bayesian trials (skipped when results already exist) | `outputs/ablation_summary.csv`, `oof_predictions_<arm>.csv` |
| 5 | Test stations | fits the winning arm on the 54 training cells and scores the 16 reserved test stations | `outputs/test_metrics_all.csv` |
| 6 | Spatial bias | Moran's I on the out-of-fold residuals, LISA when significant | I, p-value |
| 7 | Intervals | GeoCP kernel x bandwidth search, matched to the CNN-LSTM settings | `outputs/geocp_shared/geocp.json` |
| 8 | Artifact | saves the deployable model with its provenance | `outputs/artifacts/latest/` |
| 9 | Attribution | gain importance, SHAP beeswarm, single-prediction waterfall | `outputs/figures/shap_*.png` |
| 10 | Surfaces | city-wide maps for a clean day and a polluted day | `outputs/surfaces/<date>/panel_<date>.png` |
| 11 | Summary | one text block tying the numbers together | — |

---

## 2. Glossary

### Data shape

- **row** — one station-hour: a single sensor at a single hour.
- **cell** — a 500 m grid square. 84 sensors occupy 83 distinct cells (one cell holds two sensors).
- **station** — a physical sensor. 84 are ingested, 83 mappable; one sits 2.7 km outside the raster.
- **feature arm** — one feature set under test:

  | Arm | Features | Contains |
  |---|---|---|
  | `A` TIME_ONLY | 18 | cyclical time + PM2.5 history + morphology x hour interactions |
  | `B` MORPH_TIME | 15 | morphology + time + interactions, **no** PM2.5 history |
  | `C` FULL | 24 | everything |
  | `C+` | 27 | `C` + fold-aware spatial lag (k=5 neighbours) |
  | `TEXTURE` | 59 | `C` + 10 m patch statistics over a 320 m window |

### The split

- **train / calibration / test** — 54 / 13 / 16 cells. Calibration is used only to build intervals. Test is never fitted, tuned or calibrated on.
- **outer fold** — one of 3 folds that scores the model; each holds out a group of cells.
- **inner fold** — one of 3 folds inside an outer fold's training data, used to tune hyperparameters.
- **spatial CV** — cells, not hours, are held out, so no cell is seen during training and scored afterwards.

### Accuracy

- **RMSE** — root mean squared error, µg/m³. Punishes large misses hardest, so a handful of spikes dominate it.
- **MAE** — mean absolute error, µg/m³. The typical miss. Always ≤ RMSE; a large gap between them signals a heavy tail.
- **R²** — share of variance explained. 1.0 perfect, 0.0 no better than predicting the mean, negative worse than the mean.
- **IA** — Willmott's index of agreement, a bounded analogue of R².
- **bias** — mean(actual − predicted). Positive means the model under-predicts.
- **within-5 / within-10** — share of predictions within 5 / 10 µg/m³ of the observation.
- **OOF (out-of-fold)** — predictions made for cells while they were held out. The honest training-side score.
- **pooled R²** — errors summed across all rows, so large folds dominate.
- **mean R² ± sd** — per-fold scores averaged. Report this when folds differ in difficulty; pooled sits lower.
- **persistence baseline** — predict the previous hour's value. If the model cannot beat it, it is adding nothing.

### Intervals

- **coverage** — share of observations falling inside the predicted interval. Target is 90%.
- **width** — interval size in µg/m³. Narrower is better at equal coverage.
- **GeoCP** — geographically weighted conformal prediction: calibration errors are weighted by distance to the query cell.
- **split conformal** — the unweighted version, one global width.
- **LOSO** — leave-one-station-out, the rule used to choose kernel and bandwidth.

### Attribution

- **gain** — how much a feature improves tree splits, in the model's own units.
- **SHAP** — per-prediction contribution of each feature. Here in **log1p** units, because that is the scale the model was fitted on.
- **morphology share** — percentage of total attribution carried by the six morphology features. The thesis's central quantity: ~3.7% on arm `C`.

### Surfaces

- **stand-in history** — the PM2.5 series used at a cell with no sensor, built by inverse-distance weighting from the nearest reporting stations.
- **standin_coverage** — fraction of the required 73-hour history that neighbours actually reported. 1.00 means complete.
- **prediction_usable** — `True` only when a cell has **both** a stand-in history and complete morphology. Everything else is left unpainted.
- **panel** — the four snapshot hours on one shared colour scale, so hours can be compared.
- **24-hour mean** — average of the four snapshots per cell.

---

## 3. Numbers to expect

Arm `A` on the EDA-cleaned run (2026-10-21):

| Quantity | Value |
|---|---|
| rows modelled | 1,462,690 |
| cells / stations | 83 / 84 |
| OOF pooled R² | 0.565 |
| OOF mean R² ± sd | 0.605 ± 0.095 |
| OOF RMSE / MAE | 9.339 / 3.971 µg/m³ |
| test-station R² (16 stations) | 0.666 |
| persistence baseline R² | 0.584 |
| interval coverage (target 0.90) | 0.896 |
| morphology share of attribution | 3.7% |
| cells painted on a surface | 2,147 of 3,696 |

Two of these need care when quoting them:

- **0.565 vs 0.605** — pooled is row-weighted, mean is the average of folds. Folds differ a lot (0.515 / 0.705 / 0.597), so report `0.605 ± 0.095` and explain the pooled figure.
- **0.565 vs 0.666** — the first is out-of-fold on training cells (deliberately harsh), the second is the 16 held-out test stations with the full network around them. Different questions.

---

## 4. Common confusions

| Symptom | Cause |
|---|---|
| panel title says `3696 cells` and a bright wedge appears | pre-fix code: cells without history were painted with NaN-extrapolated values. Current code paints `2147` and leaves the rest blank |
| maps have no street basemap | tile cache missing and the tile server unreachable. Harmless: `basemap.configure(web_tiles=False)` is the documented offline mode |
| test R² differs from the ablation table | they are different metrics — see §3 |
| `TEXTURE` arm fails with `patch_stats.csv` missing | run `python run_pipeline.py --stage patch` |
| `C+` scores on fewer rows than other arms | rows without a qualifying spatial lag are dropped. 324,164 vs 325,070 test rows |
