"""Phase 3: score the production model on the reserved TEST stations.

Why this exists: until now the branch produced out-of-fold numbers on the *training* cells and never
scored the reserved test partition, so there was no single headline number. The CNN-LSTM run reports one
(`test` R2 0.702 on 16 stations), and without an equivalent the two branches cannot be compared.

This fits the production model on the training cells only, predicts every row of the test cells, and
writes the same metric set the CNN-LSTM run reports (RMSE, MAE, R2, Willmott's IA, Pearson r, bias,
within-5/within-10 ug/m3). Calibration cells are never used here; they belong to GeoCP.

    python -m training.final_test                 # arm C
    python -m training.final_test --arm TEXTURE

Outputs:
    outputs/test_metrics.csv        one row per arm scored
    outputs/predictions_test.csv    per-row actual/predicted/residual for this run
    outputs/test_metrics.json       the same numbers plus the provenance
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import data
from preprocessing import prep
from preprocessing import spatial_cv
from training.ablation import ARM_FEATURES, build_design_matrix

SHARED = os.path.join(data.DATA_DIR, 'fold_partitions_shared.json')

DEFAULT_PARAMS = {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8,
                  'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
                  'min_child_weight': 5}
DEFAULT_N_EST = 300


def willmott_ia(obs, pred):
    """Willmott's index of agreement: 1 is perfect, 0 is no better than the mean."""
    o = obs - obs.mean()
    p = pred - obs.mean()
    return float(1.0 - np.sum((pred - obs) ** 2) / np.sum((np.abs(p) + np.abs(o)) ** 2))


def metrics(obs, pred):
    e = obs - pred
    return {
        'rows': int(len(obs)),
        'rmse': float(np.sqrt(np.mean(e ** 2))),
        'mae': float(np.mean(np.abs(e))),
        'r2': float(1.0 - np.sum(e ** 2) / np.sum((obs - obs.mean()) ** 2)),
        'ia': willmott_ia(obs, pred),
        'pearson_r': float(np.corrcoef(obs, pred)[0, 1]),
        'bias': float(e.mean()),
        'within_5': float(np.mean(np.abs(e) <= 5.0)),
        'within_10': float(np.mean(np.abs(e) <= 10.0)),
    }


def tuned_config(arm):
    """Best outer fold's hyperparameters for `arm`, if the ablation has run for it."""
    path = data.ABLATION_RESULTS
    if os.path.exists(path):
        r = pd.read_csv(path)
        r = r[r['arm'] == arm]
        if len(r):
            row = r.set_index('outer_fold').loc[r.set_index('outer_fold')['r2'].idxmax()]
            return json.loads(row['best_params']), int(row['n_estimators']), 'ablation best fold'
    return DEFAULT_PARAMS, DEFAULT_N_EST, 'defaults (no ablation rows for this arm)'


def run(arm='C', out_dir=None, seeds=None):
    """Score the arm on the reserved test stations.

    seeds=[42, 43, 44] fits one model per seed, reports the ENSEMBLE with the per-seed
    spread, and writes the ensemble predictions. A single seed cannot say whether a
    0.66 is stable; three can.
    """
    out_dir = out_dir or data.OUT_DIR
    doc = json.load(open(SHARED, encoding='utf-8'))
    parts = doc['partitions']
    train_cells, test_cells = parts['train'], parts['test']

    df = prep.df_clean
    df_tr = df[df['Grid_ID'].isin(train_cells)].reset_index(drop=True)
    df_te = df[df['Grid_ID'].isin(test_cells)].reset_index(drop=True)
    print('arm %s: fitting on %d training cells (%s rows), scoring %d test cells (%s rows)'
          % (arm, df_tr['Grid_ID'].nunique(), f'{len(df_tr):,}',
             df_te['Grid_ID'].nunique(), f'{len(df_te):,}'))

    params, n_est, source = tuned_config(arm)
    print('  hyperparameters: %s, n_estimators=%d  [%s]' % (
        {k: round(v, 3) if isinstance(v, float) else v for k, v in params.items()}, n_est, source))

    # C+ needs the spatial lag, and it must be built from cells this fit is allowed to see: the 54
    # training cells only, never calibration or test. Without this the arm fails with
    # "['pm25_spatial_lag', ...] not in index".
    if arm == 'C+':
        # compute_spatial_lag draws neighbours from the frame it is GIVEN, so the frame must hold the
        # training rows (the only allowed sources) as well as the rows being scored. One call over the
        # union gives training cells a lag from other training cells and test cells a lag from training
        # cells -- the deployable setting, and leakage-free either way.
        allowed = sorted(int(c) for c in df_tr['Grid_ID'].unique())
        print('  computing spatial lag for C+ (k=5, %d allowed cells) ...' % len(allowed))
        both = pd.concat([df_tr, df_te], ignore_index=True)
        lag = prep.compute_spatial_lag(
            both, allowed_grid_ids=allowed,
            target_grid_ids=sorted(int(c) for c in both['Grid_ID'].unique()), k=5)
        Xtr = build_design_matrix(df_tr, arm, spatial_lag=lag)
        Xte = build_design_matrix(df_te, arm, spatial_lag=lag)
    else:
        Xtr = build_design_matrix(df_tr, arm)
        Xte = build_design_matrix(df_te, arm)
    ytr = df_tr['Value'].to_numpy(float)
    yte = df_te['Value'].to_numpy(float)

    seeds = [int(x) for x in (seeds or [42])]
    preds, seed_r2 = [], []
    for sd in seeds:
        model = xgb.XGBRegressor(n_estimators=n_est, random_state=sd, tree_method='hist',
                                 n_jobs=-1, **params)
        model.fit(Xtr, np.log1p(ytr))
        p = np.expm1(model.predict(Xte))
        preds.append(p)
        ss_res = float(((yte - p) ** 2).sum())
        ss_tot = float(((yte - yte.mean()) ** 2).sum())
        seed_r2.append(1.0 - ss_res / ss_tot)
    pred = np.mean(preds, axis=0)          # ensemble over seeds; identical to one fit when len==1
    if len(seeds) > 1:
        print('  seeds %s -> per-seed R2 %s (sd %.4f); reporting the ensemble'
              % (seeds, ' '.join('%.4f' % v for v in seed_r2), float(np.std(seed_r2, ddof=1))))

    m = metrics(yte, pred)
    m['arm'] = arm
    m['stations'] = int(df_te['Location'].nunique())
    if len(seeds) > 1:
        m['n_seeds'] = len(seeds)
        m['r2_seed_mean'] = float(np.mean(seed_r2))
        m['r2_seed_sd'] = float(np.std(seed_r2, ddof=1))

    print('\n=== TEST partition (stations the model never saw) ===')
    print('  %-24s %10s' % ('metric', 'value'))
    for k in ('rows', 'stations', 'rmse', 'mae', 'r2', 'ia', 'pearson_r', 'bias', 'within_5', 'within_10'):
        v = m[k]
        print('  %-24s %10s' % (k, v if isinstance(v, int) else '%.3f' % v))

    out = df_te[['Location', 'Grid_ID', 'DateTime', 'Value']].rename(
        columns={'Value': 'actual'}).copy()
    out['predicted'] = pred
    out['residual'] = out['actual'] - out['predicted']
    out['arm'] = arm

    # ---- per-arm files, so scoring another arm cannot destroy this one's numbers
    per_pred = os.path.join(out_dir, 'predictions_test_%s.csv' % arm)
    per_metrics = os.path.join(out_dir, 'test_metrics_%s.csv' % arm)
    out.to_csv(per_pred, index=False)
    pd.DataFrame([m]).to_csv(per_metrics, index=False)

    # ---- combined table, one row per arm (replaced in place if the arm is re-scored)
    all_csv = os.path.join(out_dir, 'test_metrics_all.csv')
    row = pd.DataFrame([m])
    if os.path.exists(all_csv):
        prev = pd.read_csv(all_csv)
        prev = prev[prev['arm'] != arm]
        row = pd.concat([prev, row], ignore_index=True)
    row = row.sort_values('r2', ascending=False).reset_index(drop=True)
    row.to_csv(all_csv, index=False)

    # ---- canonical files keep the MOST RECENT run, for figures / geocp
    out.to_csv(os.path.join(out_dir, 'predictions_test.csv'), index=False)
    pd.DataFrame([m]).to_csv(os.path.join(out_dir, 'test_metrics.csv'), index=False)
    with open(os.path.join(out_dir, 'test_metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump({'metrics': m, 'arm': arm, 'params': params, 'n_estimators': n_est,
                   'params_source': source, 'partitions': {k: len(v) for k, v in parts.items()},
                   'residual_convention': 'actual - predicted'}, fh, indent=2, default=str)

    print('\nwrote predictions_test_%s.csv, test_metrics_%s.csv' % (arm, arm))
    print('  combined test_metrics_all.csv now holds %d arm(s): %s'
          % (len(row), list(row['arm'])))
    print('  canonical test_metrics.csv / predictions_test.csv = this run (%s)' % arm)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='C', choices=['A', 'B', 'C', 'TEXTURE'])
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    run(args.arm, args.out_dir)


if __name__ == '__main__':
    main()
