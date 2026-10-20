# Experiment checklist: test every dataset, pick the best, then tune

Seven candidate datasets: `raw` (the current uncleaned OpenAQ + QC from `configs/data.yaml`) and six cleaned ones in
`data/Combined Dataset/` (`full`, `from_2023`, `from_2024`, each also as `*_with_preinstall`, which keeps the QC readings before
the sites' installation dates). Train on each, choose the best by a rule written down first, tune only that one.

## What the repo does that shapes this protocol (checked, not assumed)
- **Same stations everywhere.** The train / calibration / test split (54 / 13 / 17) depends only on the station ids and
  `data.split.seed`, so every dataset gets the identical split.
- **The CV folds are not the same.** `spatial_cv_folds` uses `GroupKFold`, which balances folds by rows per station. With
  different row counts per dataset, 31 to 36 of the 54 training stations land in a different fold in each dataset. Pooled
  out-of-fold scores are therefore confounded by which stations were held out together. `data.split.cv_assignment=by_station`
  (built) gives every dataset the same station groups.
- **The test partition is spent once.** It is excluded from fitting, hyperparameter selection and calibration. Choosing the dataset
  by test score would use it up, so **selection uses out-of-fold (CV) predictions on the training stations only.**
- **Different periods mean different rows.** Each run's own scores cover different hours, so compare runs only on the
  (location_id, timestamp) rows that all of them have.

## Phase 0: fix the rules before running anything (do not change them after seeing results)
- [ ] Arms (7): `raw`, `full`, `from_2023`, `from_2024`, `full_with_preinstall`, `from_2023_with_preinstall`, `from_2024_with_preinstall`.
- [ ] Selection metric: pooled out-of-fold RMSE on the training stations, on the rows common to all arms (MAE and R2 secondary). Never the test partition.
- [ ] Same protocol for every arm: default settings (batch 256, lr 1e-3, same architecture, early stopping patience 8, 3 CV folds).
      `data.split.seed` stays 42 (same stations); only `train.seed` varies.
- [ ] Two stages: **A** screening, 1 seed x 7 arms; **B** 3 seeds (42, 43, 44) on the best 2 to 3 arms. 7 + 6 = 13 trainings instead of 21.
- [ ] Screening runs are `preprocess` + `train` only. No `geocp` / `report`: the out-of-fold predictions come from `train`, and no test numbers exist.
- [ ] Decision rule: lowest mean RMSE over seeds. If the gap to the runner-up is within the seed-to-seed sd (or the paired CI includes 0),
      tie-break: prefer pre-installation readings removed, then the longer range.
- [ ] Pin the CV folds: add `--override "data.split.cv_assignment=by_station"` to EVERY stage command below (the default `balanced` changes the
      groups with the data). The comparison tool reports whether all runs really used the same folds.
- [ ] `git status` is clean and committed; note the commit hash (`run_info.json` records it).

## Phase 1: environment (Kaggle or local)
- [ ] Repo in a private Kaggle Dataset (no `data/`, checkpoints or `.env`) or cloned; data in private Kaggle Datasets
      (all six cleaned folders about 1.9 GB, one arm 270 to 360 MB, raster 87 MB). Check that you may upload the Quezon City data.
- [ ] `torch.cuda.is_available()` is True. Do not reinstall torch. `pip install scikit-optimize` if missing (Internet on, phone-verified account).
- [ ] `python -m pytest -q` passes there. This is the compatibility check: the dev environment is Python 3.14, pandas 3, torch 2.13.
- [ ] Timing probe: one arm with `--override train.epochs=2`; record seconds per epoch. Cost of one training is about
      (K + K/(K-1)) x epochs x seconds-per-epoch with K = 3 folds. Reported limits: about 30 GPU hours a week, 12 hours a session; `train` and `tune` cannot resume.
- [ ] Long jobs run as "Save & Run All" (commit) so the outputs persist; only `/kaggle/working` survives.
- [ ] `runs/` added to `model/.gitignore`.

## Phase 2: one folder per arm (nothing shared, nothing overwritten)
Every stage needs the same path overrides. For an arm in `$ARM` (the six cleaned ones):

```bash
python run_pipeline.py --stage preprocess --override "data.split.cv_assignment=by_station" --override "data.paths.sensor_locations_csv=../data/Combined Dataset/$ARM/locations.csv" --override "data.paths.sensor_csv_dir=../data/Combined Dataset/$ARM/sensors" --override "data.paths.processed_dir=runs/$ARM/processed" --override "data.paths.checkpoints_dir=runs/$ARM/checkpoints" --override "data.paths.snapshots_dir=runs/$ARM/snapshots" --override "data.paths.tuning_dir=runs/$ARM/tuning"
```

For `raw`, leave the two sensor paths as in `data.yaml` and override only the four `*_dir` paths.

- [ ] After `preprocess`: stations per partition are 54 / 13 / 17 in every arm; rows per partition recorded; "Sensors inside the raster: 84 of 85" is expected (Tunasan lies outside).

## Phase 3: train
```bash
python run_pipeline.py --stage train --override "data.split.cv_assignment=by_station" --override train.seed=42 --override "data.paths.sensor_locations_csv=../data/Combined Dataset/$ARM/locations.csv" --override "data.paths.sensor_csv_dir=../data/Combined Dataset/$ARM/sensors" --override "data.paths.processed_dir=runs/$ARM/processed" --override "data.paths.checkpoints_dir=runs/$ARM/checkpoints" --override "data.paths.snapshots_dir=runs/$ARM/snapshots" --override "data.paths.tuning_dir=runs/$ARM/tuning"
```
- [ ] Stage A: 7 arms x seed 42. Stage B: best arms x seeds 43 and 44 (`--override train.seed=43`).
- [ ] Each run's `run_info.json` has `"status": "completed"` and `checkpoints/<run>/oof_predictions.csv` exists; copy to the Kaggle output.
- [ ] Do not compare `SUMMARY.md` numbers across arms (different rows); they are for reading a single run.

## Phase 4: compare and choose (selection data = out-of-fold only)
```bash
python -m reporting.compare_runs --runs-root runs --out-dir runs/comparison --prefer full,from_2023,from_2024,full_with_preinstall,from_2023_with_preinstall,from_2024_with_preinstall,raw
```
  It reads `runs/<arm>/checkpoints/<run>/oof_predictions.csv` (every completed run of an arm is one seed), keeps the (station, hour) rows that
  every run has and agrees on, and writes `comparison.md`, `comparison_arms.csv`, `comparison_runs.csv`. `--prefer` is the tie-break order
  (first wins among the arms tied with the best); drop it to apply the rule by hand.
- [ ] Read the table: RMSE / MAE / R2 per arm (mean, and sd over seeds), the gap to the best arm with a 95 % station-bootstrap CI, and `Tied`
      (CI contains 0, or the gap is within the seed-to-seed sd). Only a gap whose CI excludes 0 and exceeds the seed sd is a real difference.
- [ ] Read the checks: "Same CV folds ... **yes**" (if NO, the runs were not made with `by_station`); rows dropped for disagreeing truth; the share of
      the biggest stations; "Common rows / own" per arm (an arm that keeps only a small share of its rows is being scored on a special subset).
- [ ] Sanity: does `raw` lose clearly (cleaning helps)? Do the `_with_preinstall` arms win only on their own rows? They are scored on the shared rows here.
- [ ] Apply the Phase 0 rule. Write down the chosen arm, the numbers and the reason. Do not re-run arms to get a better answer.

## Phase 5: hyperparameter tuning on the chosen arm only
- [ ] Reduce the default budget (about 20 to 25 h on CPU) with `tune.n_calls`, `tune.row_stride`, `tune.nested_evaluation`, or run on a GPU in commit mode. It cannot resume.
- [ ] Batch size is searched over 128 to 512; epochs come from early stopping.
- [ ] The outer-fold metrics in `tune_info.json` are the honest estimate of the tuned model. Keep `best_hyperparameters.json` and its `as_overrides`.

## Phase 6: final model and the single test evaluation
- [ ] `--stage train --tuned` (3 seeds) and the untuned default (same seeds) on the chosen arm.
- [ ] `geocp`, then `report`, for the final runs. Compare tuned against untuned on the same test rows: `--stage report --run-id <tuned> --baseline <untuned>`. Expect a small gain (the pilot tuning gave none).
- [ ] The test partition is read here for the first and only time. Do not iterate on it afterwards.
- [ ] `infer` with a coarse stride first (for example `--override inference.stride_px=11`), then `explain`; record the device and `chunk_cells`.

## Phase 7: record keeping
- [ ] Thesis log: arms, rules, seeds, commit hash, the dataset build (`data/Combined Dataset/README.md`), Kaggle environment versions, GPU model.
- [ ] Keep `runs_index.csv`, the comparison table, the final `SUMMARY.md` and `excluded_readings.csv`.
- [ ] State the limits: OpenAQ and QC read differently (13.2 vs 17.2 ug/m3 median network mean) and cannot be calibrated against each other;
      QC ends 2025-12-31; 2021 to 2022 have few stations reporting together; GPU runs are not bit-identical to CPU runs;
      one fixed split (54 / 13 / 17), so station-to-station variance is not averaged out; the spatial lag skips no silent station, so the
      test set is effectively 16 stations (6340868 has no rows); 2026 maps use an OpenAQ-only history (`inference.openaq_only_from_year`),
      which is farther away than in 2025 and was not seen in training; the by-year table in `SUMMARY.md` shows 2026 on OpenAQ test stations only.
- [ ] Running it on Kaggle: follow `docs/KAGGLE_GUIDE.md`.

## Built
- **B1** `data.split.cv_assignment: by_station` (default `balanced`, unchanged): every station gets its fold from a seeded shuffle of the station ids,
  independent of the rows, so every dataset validates the same station groups. Used by `train`, `tune` (outer, inner and final folds).
- **B2** `python -m reporting.compare_runs`: common rows, mean +/- sd over seeds, paired station-bootstrap CI, fold check.

## Not built (yours to do)
- **B3** the Kaggle notebook / driver that loops the arms and seeds.

## Assumed decisions (change any of them before starting)
D1 select on out-of-fold CV, not the test partition; D2 two-stage screening; D3 tie-break = pre-installation removed, then longer range;
D4 pin CV folds (`by_station`) for the comparison; D5 `raw` is included as the baseline arm; D6 screening skips `geocp` and `report`.
