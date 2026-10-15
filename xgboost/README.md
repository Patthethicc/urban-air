# XGBoost branch — street-level PM2.5 from urban morphology

Street-level PM2.5 estimation from urban morphology and temporal features, using XGBoost with nested
spatial cross-validation, Bayesian hyperparameter optimisation, a three-way ablation, and
geographically weighted conformal prediction intervals.

18 sensors · 17 grid cells · 43,273 hourly observations after cleaning · 685 days
(2024-08-16 to 2026-07-03).

> **Status.** Steps 1–5 of the work plan (`00_XGBOOST_PLAN.pdf`) are complete: data preparation,
> spatial partitions, the ablation, GeoCP, and the Deliverable 1 dataset report. Step 6 (ground
> validation) is blocked on written confirmation of the revised Deliverable 6; Step 7 (model artifact,
> manuscript text) and Step 8 (city-wide inference and maps) are not started.
>
> **Everything below is on held-out grid cells**, never on held-out rows. See `STATUS.md` for the
> running state, `STEP3_ABLATION.md` and `STEP4_GEOCP.md` for the methods records, and `FILE_GUIDE.md`
> for what every file does.

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
> stable. Quote the table above as one run of a search, not as a bit-reproducible figure — see
> `STATUS.md`.

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
- **Spatial partitions** (`fold_partitions.json`): 11 training / 3 calibration / 3 test cells, chosen by
  K-Means over cell centroids as the best of 200 seeds by partition separation. The minimum distance
  between any two partitions is 1,500 m. Partitions are assigned by cell, so all rows of a cell stay
  together. No data from a held-out cell enters cross-validation, tuning or fitting.
- **Nested spatial CV:** outer k = 3, inner k = 3, with the inner folds recomputed inside each outer
  fold so the outer test cells never influence the search. k = 3 rather than 4 because 11 training
  cells would otherwise leave 2–3 cells per outer fold.
- **Tuning:** Bayesian optimisation with **Expected Improvement** (`acq_func='EI'`, thesis Eq. 3.28),
  8 random starts, 20 calls per arm per outer fold, `n_estimators` from early stopping (cap 500,
  patience 20) averaged over inner folds. Tuned **per arm**, so no arm inherits another arm's optimum.
  Note: the search is not bit-reproducible for every arm — see the note in section 1.
- **Verification:** the spatial lag, the partitions, the ablation and the intervals each have dedicated
  checks; see section 7.

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

## 7. Verification

```
cd XGBoost
python test_spatial_lag.py             # spatial-lag regression test
python verify_ablation.py              # ablation integrity          } need the per-row files:
python adversarial_step3.py            # falsification attempt       } run `python ablation.py
python check_output_coherence.py       # output sanity               } --arms A B C --n-trials 20`
python check_morphology_coverage.py    # does the network span the raster
python verify_geocp.py                 # 28 GeoCP checks, incl. the flat-kernel limit
python test_notebook_flow.py           # runs the notebook's cells in order
```

The large per-row predictions are not committed (8.4 MB, fully regenerable); see
`XGBoost/outputs/README.md` for what is committed and the one command that regenerates the rest.

## 8. How to run

```
cd XGBoost
pip install xgboost scikit-optimize scikit-learn pandas numpy matplotlib shap

python dataset_report.py                       # Deliverable 1 dataset report
python make_fold_partitions.py                 # validate the 11 / 3 / 3 split
python ablation.py --arms A B C --n-trials 20  # ~7 min: the three-way ablation
python geocp.py --alpha 0.10                   # conformal intervals and coverage
python geocp.py --alpha 0.10 --h-mode median-distance --out-dir outputs/geocp_median_h
python verify_geocp.py
```

Or open `xgboost_pm25_pilot.ipynb` (34 cells) and **Run All** — it covers the same ground plus the
figures, including the GeoCP section. Run it top to bottom; the ablation cell takes ~7 minutes from a
clean clone. `ablation.py` must run before `geocp.py`, which reads its tuned parameters from
`outputs/ablation_results.csv`.

## 9. Layout

| Path | What |
|---|---|
| `prep_n_proc_pilot.py` | preprocessing, feature engineering, spatial lag |
| `spatial_cv.py` | cell partitions, fold maps, JSON persistence |
| `spatial_bias.py` | Moran's I / LISA on residuals |
| `ablation.py` | feature arms and nested spatial CV |
| `geocp.py` | GeoConformal Prediction |
| `dataset_report.py` | Deliverable 1 |
| `make_fold_partitions.py` | rebuilds and validates the split |
| `xgboost_pm25_pilot.ipynb` | the narrative and the figures |
| `fold_partitions.json` | the authoritative split |
| `outputs/` | results; see its own README |
| `CNN-LSTM/` | the other branch (CNN-LSTM + GeoCP), same 500 m raster and partition design |

`STATUS.md` carries the decisions and the cumulative list of changes that affect the manuscript;
`STEP3_ABLATION.md` and `STEP4_GEOCP.md` are the methods records to hand to an adviser.
