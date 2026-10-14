# outputs/ — what is committed and what is regenerated

This folder holds the XGBoost branch's results. The **large per-row files are deliberately not
committed**: they weigh 8.4 MB together and are reproducible from one command. What is here is the
aggregate results and the Deliverable 1 report.

## Committed (8 files, ~15 KB)

| File | What it is | Consumed by |
|---|---|---|
| `ablation_results.csv` | per arm × outer fold: RMSE/MAE/R², tuned hyperparameters, `n_estimators` | **`geocp.py` (required)**, `verify_ablation.py`, `adversarial_step3.py`, `check_output_coherence.py`, the notebook |
| `ablation_summary.csv` | per arm: mean ± sd over folds and pooled out-of-fold metrics | `verify_ablation.py`, the notebook |
| `dataset_report.md` | **Deliverable 1** — rows kept/dropped with a per-requirement funnel, sensors, cells, span, missingness, feature distributions | nothing (it is the deliverable) |
| `dataset_report.json` | the same numbers, machine-readable | nothing |
| `dataset_partitions.csv` | rows / cells / sensors / value stats per partition | nothing |
| `dataset_features.csv` | feature distributions across all kept rows | nothing |
| `geocp/` | GeoCP intervals and Table 4.2 coverage for the thesis-specified bandwidth search | the write-up |
| `geocp_median_h/` | the same for the geometry-only bandwidth variant | the write-up |

## Not committed, and how to regenerate

| Missing file | Size | Regenerate with |
|---|---|---|
| `oof_predictions_{A,B,C}.csv` | 8.4 MB | `python ablation.py --arms A B C --n-trials 20` (~7 min) |
| `ablation_params.json` | 2 KB | same command |
| `run_log.txt` | 7 KB | same command (stdout capture) |

`ablation.py` writes all of them and is **deterministic** — `random_state=42` and a fixed 8-point
initial design, so the regenerated files match the committed `ablation_results.csv` and
`ablation_summary.csv`.

## Consequence you should expect

Three verification scripts read the per-row predictions and will stop with a message telling you to run
the command above:

- `verify_ablation.py`
- `adversarial_step3.py`
- `check_output_coherence.py`

This is intentional. Those scripts check the results' **integrity** (no held-out cell in training, all
arms on identical rows, metrics reproducing from stored predictions), which is only meaningful against
the per-row files. They are not needed to read or cite the results.

Two things that work **without** regenerating anything:

- `python geocp.py --alpha 0.10` — it only needs `ablation_results.csv`, which is committed.
- `python dataset_report.py` — it recomputes from the input CSVs.

And one that gets slower, not broken: the notebook's ablation cell checks for the three
`oof_predictions_*.csv` files, so on a fresh clone it re-runs the ablation instead of loading from disk.
That produces a fully self-consistent state, which is arguably what you want from a fresh clone.

## Why the GeoCP folders look heavy

`geocp_*_intervals_*.csv` are **per-row** intervals for the 5,420 test rows, so each is ~1.5 MB. They are
the raw material for the coverage table and for any plot of interval widths. The summaries in
`geocp_coverage.csv` are what the write-up normally cites; the per-row files exist so the coverage
numbers can be recomputed independently. Delete them if size matters more than that check —
`geocp.py` rewrites them in about a minute.
