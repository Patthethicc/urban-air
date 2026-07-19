# XGBoost PM2.5 Pilot — 18-Sensor Run

Street-level PM2.5 estimation from urban morphology + temporal features, using XGBoost with
nested spatial k-fold cross-validation, Bayesian hyperparameter optimization, and spatial
bias detection (Moran's I). This is a **pilot-scale run** on the same 18-sensor dataset
(`Master_DL_Ready.csv`) the CNN-LSTM branch uses — not the thesis's eventual full
production-scale dataset.

## Files

| File | Role |
|---|---|
| `xgboost_pm25_pilot.ipynb` | The notebook — run this. Orchestrates everything below. |
| `prep_n_proc_pilot.py` | Data prep: spatial-joins sensors to grid cells, engineers features, builds `X`/`y`/`groups`. Runs automatically on import. |
| `spatial_cv.py` | K-Means spatial fold assignment over grid-cell centroids (outer k=4, inner k=3). |
| `spatial_bias.py` | Global & Local Moran's I on out-of-fold residuals, aggregated per grid cell. |
| `Master_DL_Ready.csv` | Source PM2.5 time series, 18 sensors, hourly, ~2-year span. |
| `full_morphology_grid.csv` | Grid-cell morphology metrics (`bh_mean`, `bh_stdev`, `bldg_dens`, `ndvi_mean`, `aspect_ratio_mean`) + cell bounding boxes. |
| `sensor_to_grid_map_full.csv` | Generated automatically by `prep_n_proc_pilot.py` on each run (sensor → grid cell join). Safe to delete; it gets rebuilt. |

**Not used by this pipeline:** `merged_small_sensors_timeseries.csv` (the original 19-sensor
pilot's raw data — unrelated to this run).

## How to run

```bash
pip install xgboost scikit-optimize shap pandas numpy scikit-learn matplotlib
jupyter notebook xgboost_pm25_pilot.ipynb
```

All five data/code files above must sit in the **same working directory** as the notebook —
everything is read/written with relative paths.

## What the prep step does differently from the original 19-sensor pilot script

1. **Spatial join, not a direct lookup.** The original `sensor_to_grid_map.csv` only covers
   the 19-sensor pilot. Here, each of the 18 sensors' projected (x, y) is matched to its
   containing cell in `full_morphology_grid.csv` and written to `sensor_to_grid_map_full.csv`
   in the same schema, so `spatial_cv.py`/`spatial_bias.py` need no changes.
2. **Time-aware lags, not row-shifts.** `Master_DL_Ready.csv`'s own `pm25_t-1/2/3` columns are
   naive row-shifts on irregularly-timed data (~20% of rows have 2–29h gaps, not 1h) — verified
   directly to mislabel stale readings as 1-hour-old during a gap. Lags/rolling stats (1–4h,
   24h, 72h) are recomputed on a per-sensor, gap-exposed hourly index instead.
3. **`street_width_mean` added** (`bh_mean / aspect_ratio_mean`) to complete the thesis's
   declared 8-feature set (Section 3.1.1), which was previously missing this one.
4. **Target is log1p-transformed** (PM2.5 is right-skewed); predictions are inverse-transformed
   before every metric below.
5. **Deliberately not added:** spatial-lag features from neighboring grid cells — the thesis
   scopes predictors to each sensor's own buffer and drops spatial identifiers by design.

## Results (last full run, `N_TRIALS=20` per outer fold)

**Data:** 43,273 rows survive gap-aware feature cleaning, across 18 sensors / 17 grid cells.

**Nested spatial CV (outer k=4):**

| Outer fold | Rows | RMSE | MAE | R² |
|---|---|---|---|---|
| 0 | 14,984 | 6.971 | 4.279 | 0.613 |
| 1 | 5,420 | 8.052 | 4.991 | 0.648 |
| 2 | 5,770 | 7.505 | 4.438 | 0.622 |
| 3 | 17,099 | 6.209 | 3.341 | 0.630 |

- **Mean: RMSE 7.185 (±0.786), MAE 4.262 (±0.686), R² 0.628 (±0.015)**
- **Pooled out-of-fold (all folds combined): RMSE 6.906, MAE 4.019, R² 0.631**

R² is stable across folds (0.613–0.648) — the model isn't just doing well on one easy region.

**Spatial bias (Moran's I on grid-level mean OOF residuals):**
Global Moran's I = **−0.125** (p = 0.472) → **not statistically significant**. Residuals appear
geographically random at this sample size; no LISA cluster follow-up was triggered.

**Production model** (refit on all data, separate k=4 spatial search):
`max_depth=5, learning_rate=0.064, subsample=0.5, colsample_bytree=0.5, reg_lambda=10, reg_alpha=0`,
204 boosting rounds.

## ⚠️ Concern: the model barely uses morphology at all

**SHAP feature attribution says the headline predictors are doing almost nothing.**
Urban-morphology features (`bh_mean`, `bh_stdev`, `bldg_dens`, `ndvi_mean`, `aspect_ratio_mean`,
`street_width_mean`) account for only **~3.6%** of total mean |SHAP|, combined:

| Feature | Mean \|SHAP\| |
|---|---|
| `pm25_lag_1` | 0.281 |
| `pm25_lag_2` | 0.125 |
| `pm25_diff_1_2` | 0.073 |
| `pm25_roll24_mean` | 0.040 |
| `hour_cos` | 0.027 |
| *(all 6 morphology features combined)* | **~0.026** |

`pm25_lag_1` *by itself* outweighs all six morphology features put together. The model is
mostly learning "PM2.5 an hour ago predicts PM2.5 now," not "urban form predicts PM2.5."

**Why this matters:** if the thesis's central claim is that 3D urban morphology (building
height, canyon geometry, vegetation, density) is a meaningful predictor of street-level PM2.5,
this run doesn't really demonstrate that — it demonstrates that PM2.5 is autocorrelated in
time, which is a much weaker and less novel claim. The strong R² below is real, but it's
largely a "PM2.5 persistence" result dressed in a morphology-features framing.

**Before writing this up as a finding, it's worth checking:**
- Refit with the `pm25_lag_*` / `pm25_roll*` / `pm25_diff_1_2` columns removed entirely, and see
  what R² and the morphology SHAP share look like on morphology + time alone. That's the real
  test of the thesis's core hypothesis.
- Whether 17 grid cells is simply too few for tree splits to find reliable morphology signal at
  all, independent of whether that signal exists physically.
- Whether CNN-LSTM has the same issue (it also receives lagged PM2.5 as input, per
  `Master_DL_Ready.csv`'s `pm25_t-1/2/3` columns) — if so, this isn't an XGBoost-specific problem,
  it's a shared framing issue across both branches.

## Are these numbers actually good?

For context, PM2.5 in this cleaned dataset (n=43,273) has **mean 16.47 µg/m³, median 14.7,
std 11.37**, ranging from 0 up to a 333 outlier.

- **R² = 0.628 under strict spatial CV** (held-out grid cells the model never saw, not just
  held-out rows) is a genuinely respectable number for this kind of small-network sensor
  problem — random/non-spatial CV would likely read noticeably higher and would be the easier,
  less honest number to report.
- **MAE = 4.26 µg/m³** is about **26% of the mean** reading — a real, usable error margin for
  a research pilot, but not tight enough for regulatory-grade point estimates (WHO's annual
  PM2.5 guideline is 5 µg/m³, so this error margin is close to the guideline value itself).
- **RMSE = 7.19 µg/m³** is proportionally larger than MAE relative to the mean, which just
  reflects the long right tail (occasional high-pollution spikes) — expected for PM2.5, not a
  red flag by itself.
- **Moran's I not significant** is a genuinely good result: it means the model's errors aren't
  systematically worse in one part of the city, which is exactly what you want before trusting
  the spatial map it produces.

**The honest overall read: numerically solid for a pilot, but not yet strong evidence for the
thesis's actual scientific claim** — see the concern above. A high R² driven mostly by
"PM2.5 now ≈ PM2.5 an hour ago" would look identical to a high R² driven by morphology, on
these metrics alone. The SHAP breakdown is what tells them apart, and right now it says the
former, not the latter.

## Known limitations

- **This is still pilot scale** — 18 sensors / 17 grid cells, all within a single
  ~10.5km × 9km morphology-raster footprint (Manila/Mandaluyong/Pasay core). The other 28
  OpenAQ sensors in the wider network fall outside that raster's coverage entirely; using them
  would require generating building-height/density/NDVI data for a larger area first (see
  prior discussion — not a code fix, a GIS data-acquisition task).
- Not directly comparable to a hypothetical full production run until CNN-LSTM and XGBoost are
  both re-evaluated on the same larger footprint.
- Low morphology SHAP share above should be interpreted cautiously given the modest number of
  distinct grid cells (17) relative to the number of morphology features (6).
