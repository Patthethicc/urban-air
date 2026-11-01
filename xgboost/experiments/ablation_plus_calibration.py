"""Ablation with the 13 GeoCP calibration cells added to training. No GeoCP anywhere.

Published result: 54 cells train the model, 13 cells calibrate the prediction intervals, and the
remaining 16 cells are scored. This experiment moves the 13 calibration cells into the training
set (67 cells), keeps the same 16 test cells, and reports point metrics only: no intervals, no
calibration stage, nothing fitted on the test cells.

The question it answers is narrow and worth asking: does the model improve when the 13 locations
that previously only set the interval width are allowed to inform the fit?

  python experiments/ablation_plus_calibration.py --arms A C --trials 20 --stride 4

Design notes
------------
1. Each calibration cell joins the outer and the inner fold of its nearest TRAINING cell, so the
   folds of the original 54 are untouched and every fold comparison stays one-to-one with the
   published run. Re-drawing folds over 67 cells would confound the comparison with a different
   spatial partition.
2. The 16 test cells are excluded from tuning, fitting and fold assignment. The only thing that
   changes between this run and the published one is how many cells the model may learn from.
3. Every row goes through `training.ablation.tune_and_evaluate`, so the search space, the log1p
   target, the metrics and early stopping are the same code paths as the published ablation.
4. Arm C+ is skipped: its spatial lag is fold-dependent and built from the fit cells of each fold,
   so extending the fold map changes what "nearest training station" means.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# data.py reads these at import time, so set them before importing anything from the pipeline.
os.environ.setdefault('XGB_DATASET', os.environ.get('XGB_DATASET', 'full'))
os.environ.setdefault('XGB_FOLDS', os.environ.get('XGB_FOLDS', 'shared'))
os.environ.setdefault('XGB_WINDOW_START', os.environ.get('XGB_WINDOW_START', '2023-01-01'))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import xgboost as xgb                   # noqa: E402

import data                             # noqa: E402
from preprocessing import prep, spatial_cv   # noqa: E402
from training import ablation as ab     # noqa: E402


def _nearest_training_cell(cell, train_cells, coords):
    """Index of the training cell closest to `cell`, in metres east / north."""
    d = np.hypot(coords.loc[train_cells, 'cx'].values - float(coords.loc[cell, 'cx']),
                 coords.loc[train_cells, 'cy'].values - float(coords.loc[cell, 'cy']))
    return int(train_cells[int(np.argmin(d))])


def extend_fold_map(doc):
    """(train_cells_with_calibration, outer_map, inner_maps) with the 54-cell folds preserved."""
    parts = doc['partitions']
    train = [int(c) for c in parts['train']]
    cal = [int(c) for c in parts['calibration']]

    outer = {int(c): int(f) for f, cells in doc['folds']['outer'].items() for c in cells}
    inner = {int(f): {int(c): int(v) for c, v in m.items()} for f, m in doc['folds']['inner'].items()}

    coords = spatial_cv.get_grid_centroids().set_index('Grid_ID')
    adopted = []
    for cell in cal:
        if cell not in coords.index:
            print('  calibration cell %s has no centroid; skipping it' % cell, flush=True)
            continue
        near = _nearest_training_cell(cell, train, coords)
        f = int(outer[near])
        outer[cell] = f
        # The inner maps hold only the FIT cells of their outer fold, so a fold's own held-out
        # cells carry no inner label. The calibration cell needs one only for the folds it is not
        # held out with; there it copies the label of the nearest cell that fold actually fits.
        for g, imap in inner.items():
            g = int(g)
            if g == f:
                continue
            pool = [int(c) for c in imap.keys()]
            if not pool:
                continue
            imap[cell] = imap[_nearest_training_cell(cell, pool, coords)]
        adopted.append((cell, near, f))

    print('  adopted %d calibration cells into training:' % len(adopted), flush=True)
    for cell, near, f in adopted:
        print('    cell %-6s -> nearest training %-6s, outer fold %d' % (cell, near, f), flush=True)
    return train + [c for c, _, _ in adopted], outer, inner


def _point_metrics(y_true, y_pred):
    err = y_true - y_pred                      # the paper's convention: actual - predicted
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        'r2': 1.0 - float((err ** 2).sum()) / ss_tot if ss_tot > 0 else float('nan'),
        'rmse': float(np.sqrt((err ** 2).mean())),
        'mae': float(np.abs(err).mean()),
        'bias': float(err.mean()),
        'within5': float((np.abs(err) <= 5).mean()),
        'within10': float((np.abs(err) <= 10).mean()),
    }


def fit_and_score_test(arm, params, n_estimators, train_cells, test_cells, df, seed=42):
    """Fit once on every training cell, score the untouched 16 test cells."""
    df_tr = df[df['Grid_ID'].isin(train_cells)]
    df_te = df[df['Grid_ID'].isin(test_cells)].reset_index(drop=True)
    X_tr = ab.build_design_matrix(df_tr, arm)
    y_tr = np.log1p(df_tr['Value'].values)
    X_te = ab.build_design_matrix(df_te, arm)

    model = xgb.XGBRegressor(n_estimators=int(n_estimators), random_state=seed,
                             tree_method='hist', n_jobs=-1, **params)
    model.fit(X_tr, y_tr)
    pred = np.expm1(model.predict(X_te))
    truth = df_te['Value'].values
    out = _point_metrics(truth, pred)
    out['n_test_rows'] = int(len(df_te))
    out['n_test_stations'] = int(df_te['Grid_ID'].nunique())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=['A', 'C'],
                    help='A, B, C or TEXTURE (C+ is skipped: fold-dependent lag)')
    ap.add_argument('--trials', type=int, default=20, help='Bayesian trials per outer fold')
    ap.add_argument('--stride', type=int, default=4, help='row stride while tuning')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=os.path.join(data.OUT_DIR, 'ablation_plus_calibration.csv'))
    args = ap.parse_args(argv)

    df = prep.df_clean.copy()
    doc = spatial_cv.load_fold_map(data.FOLD_JSON)
    parts = doc['partitions']
    test_cells = [int(c) for c in parts['test']]
    train_cells, outer_map, inner_maps = extend_fold_map(doc)

    print('\ntrain %d cells (%d + %d calibration) | test %d cells | rows %d'
          % (len(train_cells), len(parts['train']), len(parts['calibration']),
             len(test_cells), len(df)), flush=True)

    df_train = df[df['Grid_ID'].isin(train_cells)].reset_index(drop=True)
    y = np.log1p(df_train['Value'].values)
    df_train['outer_fold'] = df_train['Grid_ID'].map(outer_map)
    assert df_train['outer_fold'].notna().all(), 'a training cell has no outer fold'
    assert not set(df_train['Grid_ID']) & set(test_cells), 'test cells leaked into training'

    rows = []
    for arm in args.arms:
        if arm == 'C+':
            print('\n=== arm C+ skipped: its spatial lag is fold-dependent ===', flush=True)
            continue
        print('\n=== arm %s (%d features) ===' % (ab.ARM_LABELS.get(arm, arm),
                                                  len(ab.ARM_FEATURES[arm])), flush=True)
        X_arm = ab.build_design_matrix(df_train, arm)
        folds = sorted(df_train['outer_fold'].unique())
        fold_metrics, oofs = [], []
        for fold in folds:
            m, oof = ab.tune_and_evaluate(
                X_arm, y, df_train, fold, inner_maps[int(fold)],
                n_trials=args.trials, random_state=args.seed, n_initial_points=10,
                arm=arm, k_lag=None, row_stride=args.stride)
            fold_metrics.append(m)
            oofs.append(oof)
        oof = pd.concat(oofs, ignore_index=True)
        pooled = _point_metrics(oof['actual'].values, oof['predicted'].values)
        r2s = [m['r2'] for m in fold_metrics]
        best = max(fold_metrics, key=lambda m: m['r2'])
        params = best['best_params']
        if isinstance(params, str):
            params = json.loads(params)
        test = fit_and_score_test(arm, params, best['n_estimators'], train_cells, test_cells, df,
                                  seed=args.seed)
        row = {
            'arm': arm, 'n_features': len(ab.ARM_FEATURES[arm]),
            'n_train_cells': len(train_cells), 'n_test_cells': test['n_test_stations'],
            'cv_pooled_r2': pooled['r2'], 'cv_pooled_rmse': pooled['rmse'],
            'cv_pooled_mae': pooled['mae'],
            'cv_fold_r2_mean': float(np.mean(r2s)), 'cv_fold_r2_sd': float(np.std(r2s, ddof=1)),
            'test_r2': test['r2'], 'test_rmse': test['rmse'], 'test_mae': test['mae'],
            'test_bias': test['bias'], 'test_within5': test['within5'],
            'test_within10': test['within10'], 'trials': args.trials,
        }
        rows.append(row)
        print('    pooled CV R2 %.3f | fold R2 %.3f +- %.3f | test R2 %.3f RMSE %.2f MAE %.2f'
              % (row['cv_pooled_r2'], row['cv_fold_r2_mean'], row['cv_fold_r2_sd'],
                 row['test_r2'], row['test_rmse'], row['test_mae']), flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)      # checkpoint per arm

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print('\nwrote %s' % args.out)

    published = os.path.join(data.OUT_DIR, 'ablation_summary.csv')
    if os.path.exists(published):
        pub = pd.read_csv(published)[['arm', 'pooled_r2', 'pooled_rmse', 'pooled_mae']]
        print('\nside by side (published 54-cell CV vs 67-cell CV):')
        print(pub.merge(out[['arm', 'cv_pooled_r2', 'cv_pooled_rmse']], on='arm', how='inner')
                 .rename(columns={'pooled_r2': 'r2_54cells', 'cv_pooled_r2': 'r2_67cells',
                                  'pooled_rmse': 'rmse_54cells', 'cv_pooled_rmse': 'rmse_67cells'})
                 .round(3).to_string(index=False))
    return out


if __name__ == '__main__':
    main()
