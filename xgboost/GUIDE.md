# Reading guide

Companion to `xgboost_full.ipynb`. The notebook runs; this file explains what each cell produces, what
the figures mean, and what every term in the output refers to.

---

## 1. Cell by cell

Cell numbers are notebook order. "Expect" gives what a healthy run prints.

### Cell 0 — environment

Finds the code (a dataset holding `run_pipeline.py`, else a clone of the public repo), finds the data,
links the bundle into `data/`, installs `scikit-optimize` and `shap`.

Two details that have caused failures:

* the raster is looked up as `master_raster_v*.tif` and linked to `data/master_raster_v2.tif`, the path
  `data.py` opens. The dataset was renamed v2 → v3 once; the alias absorbs that.
* small tables are **copied**, directories and the two large files are **symlinked**. A staged run
  rewrites `fold_partitions_shared.json` and `patch_stats.csv`, and writing through a symlink into
  `/kaggle/input` fails with `EROFS`.

Expect:
```
data bundle: /kaggle/input/.../dataexample
raster linked from: .../Morphology/master_raster_v3.tif
raster: master_raster_v2.tif | exists: True
root  : /kaggle/working/urban-air/xgboost
folds : fold_partitions_shared.json
master: Master_DL_Ready_full.csv | exists: True
grid  : morphology_grid_full.csv | exists: True
```

`exists: False` on the raster means the dataset version attached to the notebook does not contain it;
section 11 then refuses to run.

### Cell 1 — data and features

`preprocessing/prep.py` runs on import: merges the 500 m morphology grid onto the sensor table, applies
the EDA cleaning, engineers gap-aware lags and cyclical time terms.

Expect `Final matrix: X=(1462690, 24)` — 1.46 M station-hours, 24 features, 83 cells.

### Cell 2 — EDA plots

Three panels: PM2.5 distribution (log axis), rows per grid cell, rows by network. The distribution has a
long right tail (0.19% of rows ≥ 100 µg/m³, max 944) — the reason RMSE ≫ MAE.

### Cell 3 — cleaning policy

Table of what the EDA removed and what is opted out (New Year masking).

### Cell 4 — dataset report

`outputs/dataset_report.md`, Deliverable 1: coverage, sensors dropped, rows kept.

### Cell 5 — spatial split

Writes `data/fold_partitions_shared.json` (54 train / 13 calibration / 16 test cells) and the 10 m patch
statistics used by the TEXTURE arm, then draws the fold map: cells coloured by outer fold, which shows
that folds are contiguous blocks of the city rather than random cells.

Expect `outer folds: {0: 24, 1: 14, 2: 16}`.

### Cell 6 — five-arm ablation

Each arm is tuned per outer fold with 20 Bayesian trials on every 4th training row (`ROW_STRIDE = 4`);
validation rows are never strided. One line per trial with inner RMSE and inner R².

Expect ~7 s/trial, ~2 min/fold, ~7 min/arm.

### Cell 4b — untuned control

Arm C with XGBoost's defaults, no search, written to `outputs/untuned/`. Its only job is to answer what
the search bought.

### Cell 7 — test stations

Fits the winning arm on the 54 training cells and scores the 16 reserved stations once per seed
(42/43/44), reporting the ensemble plus the per-seed spread.

Expect R² ≈ 0.665, MAE ≈ 3.5, within-5 ≈ 0.79.

### Cell 8 — spatial bias

Moran's I on the out-of-fold residuals, with a LISA map when significant. A near-zero I (p > 0.05) means
residuals are not spatially clustered, i.e. no region is systematically over- or under-predicted.

### Cell 9 — prediction intervals (GeoCP)

Leave-one-station-out search over kernels × bandwidths, then conformal intervals on the test stations.

Expect coverage ≈ 0.895 against a 0.90 target, width ≈ 15 µg/m³.

### Cell 10 — model artifact

`outputs/artifacts/latest/`: `model.joblib`, `run_info.json`, `features.json`. Also prints the gain
ranking, which is where the morphology share comes from.

### Cell 11 — interpretability

Gain importance, SHAP beeswarm (one point per prediction; colour = feature value), SHAP waterfall (one
prediction decomposed). SHAP values are in log1p units because that is the fitted scale.

### Cell 12 — 500 m pass

Prediction, intervals, GeoTIFF and a 2×2 panel at 500 m. Kept only as the interval source for the 10 m
product; its map is not used.

### Cell 13 — 10 m grid

`--stage grid10` builds six feature rasters from the 10 m morphology GeoTIFF (320 m windows). Then
`--stage infer10` predicts every 10 m pixel for the four snapshot hours of each date and writes a
4-band GeoTIFF per snapshot.

Expect 62–70% of 10.6 M pixels usable, ~5 min per date, panels of 1.4 MB.

### Cell 14 — verification

Scores every arm on the reserved stations, then runs `verify_ablation.py` and
`check_morphology_coverage.py`.

### Cell 15 — window sweep (off by default)

Four date windows × five arms, ~7.5 h. Off unless `RUN_WINDOW_SWEEP = True`.

### Cell 16 — summary

One block: network size, winning arm, test metrics, interval coverage, morphology share.

---

## 2. What the colours mean

**Concentration maps** use `viridis`, which runs dark purple → teal → green → yellow:

| Colour | Reading |
|---|---|
| dark purple | low on that panel's scale (clean hour, or clean side of the city) |
| teal / blue-green | around the panel median |
| green | upper-middle |
| yellow | highest values on that scale |

`viridis` is perceptually uniform (equal steps look like equal steps) and readable in greyscale and for
colour-blind readers. It is **not** an AQI palette: colour encodes concentration directly, so the same
colour at the same scale means the same µg/m³.

**The scale is shared per day, not per hour.** All four snapshots of one date use one `vmin`–`vmax`, so
morning and evening are comparable. The two dates have different scales (clean day vs polluted day) and
the title states both. Values below `vmin` clip to dark purple, which is why clean south-western pixels
look nearly black on a polluted-day panel whose floor is set by the northern plume.

**White / unpainted** is not a low value — it is no prediction. At 500 m: the cell lacks complete
morphology or a stand-in history. At 10 m: the pixel is water (Manila Bay, Laguna de Bay, the fishpond
belt), outside the building-height coverage, or has no reporting neighbour in its last two hours.

**Station dots** are white with a dark rim: white reads on both the dark and light ends of the scale and
the rim keeps them visible over yellow. They mark monitored locations, not values.

**The basemap** is lightened 72% and the prediction layer is drawn at 62% opacity, so streets and
coastlines show through without dominating. Legibility choice, not a data one.

**The AQI variant** (`*_aqi.png`) uses discrete tiers for readers who think in categories; the continuous
map is the one that preserves actual values.

---

## 3. Why some pixels have no prediction

Blank is not "clean air" — it means the pipeline has nothing to say at that pixel. Four rules, applied in
this order:

| Rule | Test | What it removes |
|---|---|---|
| **1. no morphology** | the pixel's own building height is nodata `-9999` | the strip outside the 10 m DSM coverage, mostly the northern and southern edges |
| **2. water (window)** | 320 m window mean height ≤ 0 **and** mean NDVI < 0 **and** mean building density ≤ 0.01 | Manila Bay, Laguna de Bay, the Parañaque/Las Piñas fishpond belt, La Mesa and the northern reservoirs |
| **3. water (centre pixel)** | the pixel's own height ≤ 0 **and** its NDVI < 0 | water near the shore, which rule 2 misses because its window overlaps land |
| **4. no stand-in history** | no finite neighbour value in the last two hours | pixels whose four nearest stations all missed the hour |
| gap closing | 7 × 7 morphological close on the mask | isolated DSM gaps inside built-up areas are filled, so remaining holes are at least ~70 m across |

Measured on 1.5 km windows (raster v2):

```
Manila centre        nodata 0.0% | water(window) 0.0%  | water(centre)  4.8% | valid 95.2%
Port / Intramuros    nodata 0.0% | water(window) 18.7% | water(centre) 39.9% | valid 60.1%
La Mesa reservoir    nodata 0.0% | water(window) 10.7% | water(centre) 13.8% | valid 85.6%
Manila Bay           nodata 0.0% | water(window) 99.0% | water(centre) 97.8% | valid  1.0%
```

### The two kinds of gap you will see in the figures

**A. Correct gaps — water.** Manila Bay and Laguna de Bay are 99% and mostly excluded as expected. A lake
or reservoir appears as an irregular white blob following the shoreline. Nothing is being predicted there
because the model's inputs are built from buildings and vegetation; PM2.5 over open water is not a
street-level quantity and the DSM has no height to describe.

**B. Over-aggressive gaps — paved land with a negative NDVI.** The Port of Manila and the reclamation
areas lose about 40% of their pixels to rule 3 even though they are land: large terminals and quays have
**zero building height**, and concrete or asphalt gives a **negative NDVI**, which is the same signature
as water. A reader comparing the map to a satellite image will notice these holes.

If complete land coverage matters more than a clean coastline, the honest options are:

* **drop rule 3** and accept a speckle of isolated pixels offshore (the window rule still removes open
  water);
* **use a water mask from OSM** (`natural=water`, `waterway=riverbank`) and apply it instead of the
  NDVI heuristic — the robust fix, ~30 lines in `preprocessing/patch_stack.py`;
* **keep as is** and caption it: *"water and unbuilt paved surfaces are not mapped."*

**C. Sparse single pixels — no history.** Rule 4 leaves individual pixels blank where their four nearest
stations all missed the hour. On the clean day this is ~1%, on the polluted day ~0.6%; it is why the
usable count varies slightly between the four snapshot hours of the same date.

### How to tell them apart quickly

Compare with the basemap underneath: if the basemap shows water there, the gap is rule 2/3 acting
correctly. If it shows quays, warehouses or a golf course, it is the over-aggressive case B. Numerical
breakdown for any area:

```bash
python -c "
from preprocessing import patch_stack; import numpy as np
st = patch_stack.load(); a = st['arrays']
print('valid %.1f%%' % (100*st['valid'].mean()))
print('water(window) %.1f%%' % (100*((np.nan_to_num(a['bh_mean'])<=0)&(np.nan_to_num(a['ndvi_mean'])<0)&(np.nan_to_num(a['bldg_dens'])<=0.01)).mean()))
"
```

---

## 4. Terms

### Data shape

* **row** — one station-hour.
* **cell** — a grid square. 3,696 at 500 m; 10,634,913 pixels at 10 m.
* **station** — a physical sensor. 84 are ingested, 83 map to distinct 500 m cells (cell 1749 holds two).
* **arm** — a feature set: `A` time+history (18), `B` morphology+time (15), `C` everything (24),
  `C+` `C` plus a spatial lag (27), `TEXTURE` `C` plus 10 m patch statistics (59).

### Split and sampling

* **train / calibration / test** — 54 / 13 / 16 cells. Calibration builds intervals only; test is never
  fitted, tuned or calibrated on.
* **outer fold** — one of three held-out cell groups used for scoring. **inner fold** — one of three used
  inside a fold for tuning.
* **OOF (out-of-fold)** — predictions made for cells while they were held out.
* **N trials** — Bayesian optimisation calls per fold. **row stride** — tuning uses every Nth training
  row, validation rows always complete. **seed ensemble** — the same fit under three seeds, averaged.

### Accuracy

* **RMSE** — root mean squared error, µg/m³; dominated by large misses.
* **MAE** — mean absolute error, the typical miss. Always ≤ RMSE; the gap measures the tail's weight.
* **R²** — share of variance explained. 0 = no better than the mean, negative = worse.
* **IA** — Willmott's index of agreement, a bounded analogue of R².
* **bias** — mean(actual − predicted); positive means under-prediction.
* **within-5 / within-10** — share of predictions within 5 / 10 µg/m³.
* **pooled R²** — errors summed over all rows, so large folds dominate.
* **mean R² ± sd** — per-fold scores averaged; report this when folds differ in difficulty.
* **persistence** — predict the previous hour's value; the baseline any model must beat.

### Intervals

* **coverage** — share of observations inside the interval; target 90%.
* **width** — interval size in µg/m³; narrower is better at equal coverage.
* **GeoCP** — geographically weighted conformal prediction: calibration errors weighted by distance to
  the query location.
* **bandwidth / k** — the weighting kernel's scale, chosen by leave-one-station-out.
* **LOSO** — leave-one-station-out; the check that a bandwidth generalises.
* **conformal quantile (q)** — the calibrated residual quantile that sets the width.

### Attribution

* **gain** — a feature's total split improvement in the model's units.
* **SHAP** — per-prediction contribution, here in log1p units; positive pushes the prediction up.
* **morphology share** — percentage of attribution carried by the six morphology features. Only
  meaningful on an arm that contains them (`C`, `TEXTURE`); arm `A` scores 0 by construction.

### Surfaces

* **stand-in history** — PM2.5 at a cell with no sensor, built by inverse-distance weighting from the
  nearest reporting stations (k = 4).
* **standin_coverage** — fraction of required hours that neighbours actually reported; 1.00 is complete.
* **prediction_usable** — `True` only when the cell has valid morphology *and* a stand-in history.
* **patch stack** — the six morphology features as 320 m box statistics centred on every 10 m pixel,
  stored as rasters rather than a table.
* **window sweep** — the ablation repeated on different date ranges, to test whether a conclusion
  survives the choice of window.

---

## 5. Numbers from the last verified run

Raster v2, 20 trials, stride 1:

| Quantity | Value |
|---|---|
| rows modelled | 1,462,690 |
| cells / stations | 83 / 84 |
| OOF pooled R² (arm A) | 0.566 |
| OOF mean R² ± sd | 0.605 ± 0.094 |
| test R², C+ / A / C / TEXTURE / B | 0.671 / 0.665 / 0.660 / 0.659 / −0.033 |
| interval coverage | 0.895 (target 0.90) |
| 500 m cells painted | 2,147 of 3,696 |
| 10 m pixels painted | ~6.6 M of 10.6 M (62%) |

The raster was later revised to **v3**, whose height and aspect bands differ (texture medians moved from
3.86 to 4.63 m and 0.84 to 1.24). Runs after that date are a fresh baseline on v3 and are not directly
comparable to the table above. Record which raster version produced which numbers.

Three numbers are often confused:

* **0.566 vs 0.605** — pooled is row-weighted; 0.605 is the average of folds (0.518 / 0.704 / 0.594).
* **0.566 vs 0.665** — out-of-fold on training cells versus the 16 held-out stations.
* **morphology 0% (arm A) vs 4.3% (arm C)** — arm A contains no morphology features, only interactions.

---

## 6. Common confusions

| Symptom | Cause |
|---|---|
| panel says `3696 cells` and a bright wedge appears | pre-fix code painted cells without history; current code leaves them blank |
| `raster ... exists: False` | dataset version attached to the notebook lacks `Morphology/master_raster_v*.tif` |
| `EROFS` writing `data/fold_partitions_shared.json` | a symlink was written through; small tables must be copied |
| maps have no basemap | tile cache missing and the tile server unreachable; harmless, `XGB_NO_TILES=1` silences it |
| test R² ≠ ablation R² | different quantities: held-out stations vs out-of-fold cells |
| `TEXTURE` fails with `patch_stats.csv` missing | run `--stage patch` |
| 10 m summary shows a 4-hour total | older `grid10m.py`; the current one writes per-hour min/max |
| 10 m panel has black specks | pixels below that panel's shared colour floor, usually pond edges |
| a hole in the middle of the city | paved land with height 0 and negative NDVI (ports, reclamation, golf courses) hits the centre-pixel water rule |
| an irregular white blob near the northern edge | a reservoir; excluded correctly by the window water rule |
