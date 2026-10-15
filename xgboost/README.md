# outputs/ — what is committed and what is regenerated

This folder holds the XGBoost branch's results. **The large per-row files are deliberately not
committed** (8.4 MB, reproducible from one command). What is here is the aggregate results, the
Deliverable 1 report, the Step 7 model artifact, the Step 8 surfaces, and the Step 4 intervals.

## Committed

| File | What it is | Consumed by |
|---|---|---|
| `ablation_results.csv` | per arm x outer fold: RMSE/MAE/R², tuned hyperparameters, `n_estimators` | **`geocp.py` (required)**, `run_pipeline.py --stage save` and `--stage infer`, `verify_ablation.py`, `adversarial_step3.py`, `check_output_coherence.py`, the notebook |
| `ablation_summary.csv` | per arm: mean ± sd over folds and pooled out-of-fold metrics | `verify_ablation.py`, the notebook |
| `dataset_report.md` | **Deliverable 1** — rows kept/dropped with a per-requirement funnel, sensors, cells, span, missingness, feature distributions | nothing (it is the deliverable) |
| `dataset_report.json` | the same numbers, machine-readable | nothing |
| `dataset_partitions.csv` | rows / cells / sensors / value stats per partition | nothing |
| `dataset_features.csv` | feature distributions across all kept rows | nothing |
| `artifacts/<run_id>/` | **Step 7**: `model.joblib` + `run_info.json` + `features.json` | `inference/cell_inference.py` |
| `surfaces/<date>/` | **Step 8**: per-snapshot per-cell CSV, AQI map PNG, summary JSON | the write-up |
| `geocp/` | GeoCP intervals and Table 4.2 coverage, thesis-specified bandwidth search | the write-up |
| `geocp_median_h/` | the same for the geometry-only bandwidth variant | the write-up |

### Not committed, and how to regenerate

| Missing file | Size | Regenerate with |
|---|---|---|
| `oof_predictions_{A,B,C}.csv` | 8.4 MB | `python run_pipeline.py --stage ablation` (~7–9 min) |
| `ablation_params.json` | 2 KB | same command |

`run_log.txt` used to be listed here. **Nothing writes it** — it is not produced by any script in the
repository, so it was removed from this list.

### The ablation is NOT bit-reproducible — do not assume the regenerated files will match

`ablation.py` sets `random_state=42` throughout, but **arms A can move between runs**. Measured across
two runs of the identical command: **arm A** pooled R² **0.638 → 0.630** (and different `best_params` in
outer folds 1–2), while **arms B and C reproduced exactly** to three decimals. Cause: scikit-optimize
updates its surrogate as each objective returns while the XGBoost fits run with `n_jobs=-1`, so parallel
completion order varies and the search can settle on a different optimum.

Consequences to keep in mind:

- The **conclusion is unaffected** — in both runs arm B scores below a constant and arm A matches arm C.
- **`geocp.py` is unaffected** — it reads arm **C**, whose parameters and `n_estimators` reproduce
  exactly, so the Step 4 numbers are stable.
- Quote the committed `ablation_results.csv` as **one run of a stochastic search**, not as a
  bit-reproducible figure. Setting `n_jobs=1` in the fits would make it reproducible, at a large speed
  cost; that was not done.

`n_trials` must be **20** to match the committed results. `training/ablation.py` defaults to 20;
the notebook's ablation cell is set to 20 for the same reason. At 16 the tuning searches less and the
numbers will not match what is in this folder.

## Consequence you should expect

Three verification scripts read the per-row predictions and will stop with one actionable message
telling you to run `--stage ablation` first:

- `checks/verify_ablation.py`
- `checks/adversarial_step3.py`
- `checks/check_output_coherence.py`

This is intentional. Those scripts check the results' **integrity** (no held-out cell in training, all
arms on identical rows, metrics reproducing from stored predictions), which is only meaningful against
the per-row files. They are not needed to read or cite the results.

These work **without** regenerating anything:

- `run_pipeline.py --stage geocp` — only needs `ablation_results.csv`, which is committed.
- `run_pipeline.py --stage report` — recomputes from the input CSVs.
- `run_pipeline.py --stage save` / `--stage infer` — need only `ablation_results.csv` for the production
  model's hyperparameters, not the per-row files.

And one that gets slower, not broken: the notebook's ablation cell checks for the three
`oof_predictions_*.csv` files, so on a fresh clone it re-runs the ablation instead of loading from disk.

## Why the GeoCP folders look heavy

`geocp_*_intervals_*.csv` are **per-row** intervals for the 5,420 test rows, so each is ~1.5 MB. They are
the raw material for the coverage table and for any plot of interval widths. The summaries in
`geocp_coverage.csv` are what the write-up normally cites; the per-row files exist so the coverage
numbers can be recomputed independently. Delete them if size matters more than that check —
`geocp.py` rewrites them (about 10 minutes, since `quantile_grid` is exact and the bandwidth search
calls it 5,420 times).
