import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""STEP 2 runner: rebuild the train / calibration / test split and the fold maps.

Run from the XGBoost folder:

    python make_fold_partitions.py

Writes fold_partitions.json (the split every later step reads) and prints a validation report:
per-partition row counts, the spatial-lag coverage that coverage implies, and a leakage test that
corrupts a held-out cell's own readings to prove its lag value does not depend on them.
"""
import json

import numpy as np
import pandas as pd

import data
from preprocessing import prep
from preprocessing import spatial_cv

FOLD_JSON = data.FOLD_JSON
K_OUTER = 3   # plan Section 3.2: 17 cells -> 11 training cells, so k=4 would leave ~2-3 cells/fold
K_INNER = 3
# Calibration/test sizes. `None` derives them from the cell count as max(3, round(0.15 * cells)), so
#   pilot (17 cells): max(3, 2.55) = 3  -> 11 / 3 / 3, exactly the published split
#   full  (83 cells): max(3, 12.45) = 12 -> ~59 / 12 / 12, a sane train-heavy split
# Hardcoding the pilot's 3/3 while leaving the train target at 11 gave train=10 vs calibration=34 and
# test=39 on the full dataset -- `search_partitions` clusters for separation, not size, so the sizes have
# to be stated relative to the network.
N_CALIBRATION = None
N_TEST = None
CALIBRATION_FRACTION = 0.15
TEST_FRACTION = 0.15
MIN_PARTITION = 3
# On the pilot's 17 cells the k=3 clustering happens to give the published 11/3/3, so `balanced` must
# stay False there to reproduce it exactly. On a larger network k=3 cannot honour balanced sizes (it
# yielded 39/10/34 when 59/12/12 was asked for), so `balanced` switches on automatically.
BALANCED = data.DATASET == 'full'
K_LAG = 4     # matches the k=4 used by build_knn_weights in spatial_bias.py


def coverage_table(df, lag, label):
    """How often the lag actually has a value, per partition, joined back on (Grid_ID, DateTime)."""
    merged = df.merge(lag[['Grid_ID', 'DateTime', 'pm25_spatial_lag', 'pm25_spatial_lag_w',
                           'pm25_spatial_lag_n']], on=['Grid_ID', 'DateTime'], how='left')
    return {
        'label': label,
        'rows': int(len(merged)),
        'lag_present': float(merged['pm25_spatial_lag'].notna().mean()),
        'median_contributors': float(merged['pm25_spatial_lag_n'].median()),
        'mean_contributors': float(merged['pm25_spatial_lag_n'].mean()),
        '_merged': merged,
    }


def main(n_calibration=None, n_test=None, force=False):
    # The shared split is ADOPTED from the CNN-LSTM run (training/shared_split.py), not produced by
    # clustering. Re-running this stage under XGB_FOLDS=shared would silently replace the 54/13/16
    # station partition with a K-Means one, destroying comparability. Refuse unless forced.
    if not force and data.FOLD_JSON.endswith('fold_partitions_shared.json'):
        raise SystemExit(
            'refusing to overwrite the SHARED split.\n'
            '  data/fold_partitions_shared.json is adopted from the CNN-LSTM run to keep the two\n'
            '  branches comparable; this stage would replace it with a K-Means partition.\n'
            '  Use `python run_pipeline.py --stage split` to regenerate it, or clear XGB_FOLDS\n'
            '  (and XGB_FOLDS in .env) to work on this branch\'s own split instead.')

    df = prep.df_clean

    n_cells = int(df['Grid_ID'].nunique())
    if n_calibration is None:
        n_calibration = N_CALIBRATION if N_CALIBRATION is not None else \
            max(MIN_PARTITION, int(round(CALIBRATION_FRACTION * n_cells)))
    if n_test is None:
        n_test = N_TEST if N_TEST is not None else \
            max(MIN_PARTITION, int(round(TEST_FRACTION * n_cells)))
    n_train = n_cells - n_calibration - n_test
    if n_train < K_OUTER * K_INNER:
        raise SystemExit(
            f'{n_cells} cells with calibration={n_calibration} / test={n_test} leaves {n_train} training '
            f'cells, fewer than outer k={K_OUTER} x inner k={K_INNER} = {K_OUTER * K_INNER}. '
            f'Lower --n-calibration / --n-test.')
    print(f'  cells={n_cells} -> train={n_train} / calibration={n_calibration} / test={n_test}')

    doc = spatial_cv.build_fold_document(k_outer=K_OUTER, k_inner=K_INNER,
                                         n_calibration=n_calibration, n_test=n_test,
                                         n_train=n_train, balanced=BALANCED)
    spatial_cv.save_fold_map(doc, FOLD_JSON)
    reloaded = spatial_cv.load_fold_map(FOLD_JSON)

    parts = doc['partitions']
    print('=' * 78)
    print('PARTITIONS (written to %s)' % FOLD_JSON)
    print('=' * 78)
    for name in spatial_cv.PARTITION_NAMES:
        cells = parts[name]
        rows = int(df['Grid_ID'].isin(cells).sum())
        print(f"  {name:12s} {len(cells):2d} cells  {rows:6d} rows   {cells}")
    print(f"  min distance between partitions: {doc['info']['min_distance_between_partitions_m']} m")
    print(f"  outer k={K_OUTER}: {doc['folds']['outer']}")
    print(f"  inner k={K_INNER}: {doc['folds']['inner']}")
    print(f"  JSON round-trip OK: {reloaded['partitions'] == parts}")

    # ---------------------------------------------------------------- spatial lag per partition
    print()
    print('=' * 78)
    print('SPATIAL LAG (k=%d, inverse distance, own cell always excluded)' % K_LAG)
    print('=' * 78)
    train_cells = parts['train']
    rows = []
    for name in spatial_cv.PARTITION_NAMES:
        cells = parts[name]
        # One call per partition: the full df_clean as the neighbour pool, restricted to `allowed`,
        # and lag rows produced only for this partition's cells.
        lag = prep.compute_spatial_lag(df, allowed_grid_ids=train_cells, target_grid_ids=cells,
                                       k=K_LAG)
        cov = coverage_table(df[df['Grid_ID'].isin(cells)], lag, name)
        rows.append(cov)
        print(f"  {name:12s} rows={cov['rows']:6d}  lag present={cov['lag_present']:6.1%}  "
              f"median contributors={cov['median_contributors']:.0f}  "
              f"mean contributors={cov['mean_contributors']:.2f}")

    # ------------------------------------------------------- leakage test (corruption invariant)
    print()
    print('=' * 78)
    print('LEAKAGE TEST: replace each held-out cell\'s OWN readings with 9999 and re-lag it.')
    print('If the lag is fold-aware, the held-out lag values must be identical.')
    print('=' * 78)
    ok = True
    for name in ('test', 'calibration'):
        cells = parts[name]
        base = prep.compute_spatial_lag(df, allowed_grid_ids=train_cells, target_grid_ids=cells,
                                        k=K_LAG)
        corrupt = df.copy()
        corrupt.loc[corrupt['Grid_ID'].isin(cells), 'Value'] = 9999.0
        dirty = prep.compute_spatial_lag(corrupt, allowed_grid_ids=train_cells,
                                         target_grid_ids=cells, k=K_LAG)
        same = base['pm25_spatial_lag'].equals(dirty['pm25_spatial_lag'])
        ok &= same
        print(f"  {name:12s} identical after self-corruption: {same}")
    print(f"  PASS: no held-out cell's lag depends on its own readings: {ok}")

    # ------------------------------------------------ contributors can never exceed k, or be self
    print()
    print('=' * 78)
    print('SANITY CHECKS')
    print('=' * 78)
    lag_all = prep.compute_spatial_lag(df, allowed_grid_ids=train_cells, k=K_LAG)
    print('  contributor count never exceeds k:',
          bool(lag_all['pm25_spatial_lag_n'].max() <= K_LAG),
          '(max=%d)' % lag_all['pm25_spatial_lag_n'].max())
    print('  lag rows with a value globally: %.1f%%' %
          (100 * lag_all['pm25_spatial_lag'].notna().mean()))
    print('  lag rows with A value but zero weight (impossible):',
          int(((lag_all['pm25_spatial_lag'].notna()) & (lag_all['pm25_spatial_lag_w'] == 0)).sum()))

    # the same lag computed twice must be identical (no hidden randomness)
    again = prep.compute_spatial_lag(df, allowed_grid_ids=train_cells, k=K_LAG)
    print('  deterministic across runs:', lag_all['pm25_spatial_lag'].equals(again['pm25_spatial_lag']))

    # partition overlap must be empty
    sets = {n: set(parts[n]) for n in spatial_cv.PARTITION_NAMES}
    print('  partitions are disjoint:',
          not (sets['train'] & sets['test']) and not (sets['train'] & sets['calibration'])
          and not (sets['test'] & sets['calibration']))
    print('  partitions cover all %d cells:' % df['Grid_ID'].nunique(),
          len(sets['train'] | sets['test'] | sets['calibration']) == df['Grid_ID'].nunique())

    with open(FOLD_JSON, encoding='utf-8') as fh:
        print('  %s is valid JSON, top-level keys: %s' % (FOLD_JSON, sorted(json.load(fh))))


if __name__ == '__main__':
    main()
