"""Cohort-aware sensitivity run: train and score one arm inside a date window.

The EDA report recommends restricting cross-station (spatial-lag) work to the period where the network is
at full strength -- "or restrict joint multi-station training windows to April 2024-2025". The converted
long files actually begin the third QC cohort on 2024-01-01 PHT, so this defaults to that boundary.

The point of running it as a separate experiment is that the comparison stays honest: the window applies
to BOTH the lag arm and its aspatial counterpart, so C+ (windowed) is compared with A (windowed) on
identical rows rather than against an all-years baseline.

    python experiments/cohort_window.py --arm C+ --window-start 2024-01-01
    python experiments/cohort_window.py --arm A  --window-start 2024-01-01
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', default='C+', choices=['A', 'B', 'C', 'C+', 'TEXTURE'])
    ap.add_argument('--window-start', default='2024-01-01')
    ap.add_argument('--dump', default=None)
    a = ap.parse_args()

    os.environ['XGB_WINDOW_START'] = a.window_start

    import data
    from preprocessing import prep, spatial_cv
    from training.ablation import ARM_FEATURES, build_design_matrix
    from training.final_test import tuned_config

    doc = spatial_cv.load_fold_map()
    parts = doc['partitions']
    df = prep.df_clean
    df_tr = df[df['Grid_ID'].isin(parts['train'])].reset_index(drop=True)
    df_te = df[df['Grid_ID'].isin(parts['test'])].reset_index(drop=True)
    if len(df_tr) == 0 or len(df_te) == 0:
        raise SystemExit('window leaves no rows: %s' % a.window_start)

    params, n_est, src = tuned_config(a.arm)
    ytr = df_tr['Value'].to_numpy(float)
    yte = df_te['Value'].to_numpy(float)

    if a.arm == 'C+':
        allowed = sorted(int(c) for c in df_tr['Grid_ID'].unique())
        both = pd.concat([df_tr, df_te], ignore_index=True)
        lag = prep.compute_spatial_lag(
            both, allowed_grid_ids=allowed,
            target_grid_ids=sorted(int(c) for c in both['Grid_ID'].unique()), k=5)
        Xtr = build_design_matrix(df_tr, a.arm, spatial_lag=lag)
        Xte = build_design_matrix(df_te, a.arm, spatial_lag=lag)
    else:
        lag = None
        Xtr = build_design_matrix(df_tr, a.arm)
        Xte = build_design_matrix(df_te, a.arm)

    model = xgb.XGBRegressor(n_estimators=n_est, random_state=42, tree_method='hist',
                             n_jobs=-1, **params)
    model.fit(Xtr, np.log1p(ytr))
    pred = np.expm1(model.predict(Xte))
    err = yte - pred

    out = {
        'arm': a.arm, 'window_start': a.window_start,
        'train_rows': int(len(df_tr)), 'test_rows': int(len(df_te)),
        'train_cells': int(df_tr['Grid_ID'].nunique()), 'test_cells': int(df_te['Grid_ID'].nunique()),
        'features': len(ARM_FEATURES[a.arm]),
        'rmse': float(np.sqrt(np.mean(err ** 2))), 'mae': float(np.mean(np.abs(err))),
        'r2': float(1 - np.sum(err ** 2) / np.sum((yte - yte.mean()) ** 2)),
        'bias': float(err.mean()),
        'within_5': float(np.mean(np.abs(err) <= 5.0)),
        'within_10': float(np.mean(np.abs(err) <= 10.0)),
        'params_source': src,
    }
    if lag is not None:
        out['lag_present_pct'] = float(100 * lag['pm25_spatial_lag'].notna().mean())
    print(json.dumps(out))

    if a.dump:
        pd.DataFrame({'Location': df_te['Location'].values, 'DateTime': df_te['DateTime'].values,
                      'actual': yte, 'predicted': pred}).to_csv(a.dump, index=False)


if __name__ == '__main__':
    main()
