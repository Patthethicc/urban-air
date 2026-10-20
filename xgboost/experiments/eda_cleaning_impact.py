"""Measure what the EDA cleaning actually changes, on one fold with fixed hyperparameters.

The EDA recommends dropping sensor-floor readings (<= 0.5 ug/m3) and masking the New Year's Eve fireworks
window. Both are tiny in row count (0.45% together) but the tail they remove is where the squared error
lives, so the effect on RMSE/R2 is not proportional to the rows removed. This runs the same arm on the
same fold with cleaning on and off and prints the difference, so the thesis can state it rather than guess.

    python experiments/eda_cleaning_impact.py --arm A --fold 0 --clean on
    python experiments/eda_cleaning_impact.py --arm A --fold 0 --clean off
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='A')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--clean', default='on', choices=['on', 'off'])
    ap.add_argument('--dump', default=None, help='write per-row predictions here for a paired comparison')
    a = ap.parse_args()

    os.environ['XGB_CLEAN'] = a.clean

    import data
    from preprocessing import prep, spatial_cv
    from training.ablation import ARM_FEATURES, build_design_matrix
    from training.final_test import tuned_config

    df = prep.df_clean
    doc = spatial_cv.load_fold_map()
    df_tr = df[df['Grid_ID'].isin(doc['partitions']['train'])].reset_index(drop=True)
    outer = {int(c): int(f) for f, cs in doc['folds']['outer'].items() for c in cs}
    df_tr['outer_fold'] = df_tr['Grid_ID'].map(outer)

    params, n_est, _ = tuned_config(a.arm)
    te = (df_tr['outer_fold'] == a.fold).to_numpy()
    X = build_design_matrix(df_tr, a.arm)
    ytr = df_tr.loc[~te, 'Value'].to_numpy(float)
    yte = df_tr.loc[te, 'Value'].to_numpy(float)

    model = xgb.XGBRegressor(n_estimators=n_est, random_state=42, tree_method='hist',
                             n_jobs=-1, **params)
    model.fit(X[~te], np.log1p(ytr))
    pred = np.expm1(model.predict(X[te]))
    err = yte - pred

    out = {
        'cleaning': a.clean, 'arm': a.arm, 'fold': a.fold,
        'rows': int(te.sum()), 'features': len(ARM_FEATURES[a.arm]),
        'rmse': float(np.sqrt(np.mean(err ** 2))), 'mae': float(np.mean(np.abs(err))),
        'r2': float(1 - np.sum(err ** 2) / np.sum((yte - yte.mean()) ** 2)),
        'bias': float(err.mean()),
        'max_actual': float(yte.max()), 'rows_gt_150': int((yte > 150).sum()),
        'rmse_excl_gt150': float(np.sqrt(np.mean(err[yte <= 150] ** 2))),
    }
    if a.dump:
        pd.DataFrame({'Location': df_tr.loc[te, 'Location'].values,
                      'DateTime': df_tr.loc[te, 'DateTime'].values,
                      'actual': yte, 'predicted': pred}).to_csv(a.dump, index=False)
        print('wrote', a.dump, file=__import__('sys').stderr)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
