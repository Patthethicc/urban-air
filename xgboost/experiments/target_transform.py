"""Phase 0: does the target transform decide whether we beat persistence?

Motivation (measured, not assumed): on the full network the current model (log1p target, MSE) scores
R2 0.588 while naive 1-hour persistence scores 0.596 -- i.e. the model is *worse than copying the
previous hour*. The CNN-LSTM run beats persistence (0.702 vs 0.644) and it standardises the target
instead. On a right-skewed variable, log1p+MSE optimises relative error while RMSE is scored on the raw
scale, so peaks are under-fitted -- and the top 1% of hours carry 61% of the squared error.

This trains the SAME model on the SAME rows and fold with three target transforms and scores all of
them on the raw ug/m3 scale, against the persistence baseline.

    python experiments/target_transform.py            # outer fold 0, arm C features
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data                      # noqa: E402
from preprocessing import prep    # noqa: E402
from preprocessing import spatial_cv  # noqa: E402
from training.ablation import ARM_FEATURES, build_design_matrix  # noqa: E402

import xgboost as xgb             # noqa: E402

ARM = 'C'
FOLD = 0
N_TRIALS_PARAMS = None            # reuse the tuned params stored by the last ablation


def metrics(y, p):
    e = y - p
    ss = np.sum((y - y.mean()) ** 2)
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))),
            'r2': float(1 - np.sum(e ** 2) / ss), 'bias': float(e.mean())}


def main():
    df = prep.df_clean
    doc = spatial_cv.load_fold_map()
    parts = doc['partitions']
    train_cells = list(parts['train'])

    df_train = df[df['Grid_ID'].isin(train_cells)].reset_index(drop=True)
    outer_map = {int(c): int(f) for f, cells in doc['folds']['outer'].items() for c in cells}
    df_train['outer_fold'] = df_train['Grid_ID'].map(outer_map)

    res = pd.read_csv(data.ABLATION_RESULTS)
    row = res[(res['arm'] == ARM) & (res['outer_fold'] == FOLD)].iloc[0]
    params = json.loads(row['best_params'])
    n_est = int(row['n_estimators'])
    print('arm %s, outer fold %d, params from the last ablation (n_estimators=%d)'
          % (ARM, FOLD, n_est))
    print('  ', {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()})

    X = build_design_matrix(df_train, ARM)
    te = (df_train['outer_fold'] == FOLD).to_numpy()
    Xtr, Xte = X[~te], X[te]
    y_raw_tr = df_train.loc[~te, 'Value'].to_numpy(float)
    y_raw_te = df_train.loc[te, 'Value'].to_numpy(float)

    print('\n  train rows %s | test rows %s | test cells %d'
          % (f'{(~te).sum():,}', f'{te.sum():,}', df_train.loc[te, 'Grid_ID'].nunique()))

    out = {}

    # ---------------- persistence baseline on the SAME test rows
    d = df_train.loc[te, ['Location', 'DateTime', 'Value']].sort_values(['Location', 'DateTime'])
    prev = d.groupby('Location')['Value'].shift(1)
    ok = prev.notna().to_numpy()
    out['persistence (t-1)'] = metrics(y_raw_te[ok], prev.to_numpy()[ok])

    # ---------------- the three transforms
    variants = {
        'log1p (current)': (lambda v: np.log1p(v), lambda p: np.expm1(p)),
        'raw': (lambda v: v, lambda p: p),
        'standardised': None,   # handled below (needs the train mean/sd)
    }

    for name, pair in variants.items():
        if name == 'standardised':
            mu, sd = y_raw_tr.mean(), y_raw_tr.std()
            ytr = (y_raw_tr - mu) / sd
            inv = lambda p, mu=mu, sd=sd: p * sd + mu
        else:
            fwd, inv = pair
            ytr = fwd(y_raw_tr)
        model = xgb.XGBRegressor(n_estimators=n_est, random_state=42, tree_method='hist',
                                 n_jobs=-1, **params)
        model.fit(Xtr, ytr)
        pred = inv(model.predict(Xte))
        out[name] = metrics(y_raw_te, pred)
        print('  %-18s done' % name)

    print('\n=== raw-scale metrics on outer fold %d (%s test rows) ===' % (FOLD, f'{te.sum():,}'))
    print('  %-20s %8s %8s %8s %8s' % ('predictor', 'RMSE', 'MAE', 'R2', 'bias'))
    for k, v in out.items():
        print('  %-20s %8.3f %8.3f %8.3f %+8.3f' % (k, v['rmse'], v['mae'], v['r2'], v['bias']))

    best = max(out, key=lambda k: out[k]['r2'])
    print('\n  best by R2: %s' % best)
    print('  persistence bar to clear: R2 %.3f / RMSE %.3f'
          % (out['persistence (t-1)']['r2'], out['persistence (t-1)']['rmse']))
    json.dump(out, open(os.path.join(data.OUT_DIR, 'target_transform.json'), 'w'), indent=2)
    print('\nwrote outputs/target_transform.json')


if __name__ == '__main__':
    main()
