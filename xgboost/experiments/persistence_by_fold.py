"""Phase 0b: model vs persistence, per fold (pooling across folds inflates R2).

The pooled OOF frame mixes three folds whose test cells have different mean concentrations. R2 uses the
global mean as its baseline, so pooling adds between-fold variance and inflates R2 for ANY predictor --
which is why a pooled comparison said "model worse than persistence" while fold 0 alone says the
opposite. The honest comparison is inside each fold, then averaged.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data                      # noqa: E402
from preprocessing import prep    # noqa: E402
from preprocessing import spatial_cv  # noqa: E402
from training.ablation import build_design_matrix  # noqa: E402

ARM = 'C'


def metrics(y, p):
    e = y - p
    ss = np.sum((y - y.mean()) ** 2)
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))),
            'r2': float(1 - np.sum(e ** 2) / ss), 'bias': float(e.mean())}


def main():
    df = prep.df_clean
    doc = spatial_cv.load_fold_map()
    df_train = df[df['Grid_ID'].isin(doc['partitions']['train'])].reset_index(drop=True)
    outer_map = {int(c): int(f) for f, cells in doc['folds']['outer'].items() for c in cells}
    df_train['outer_fold'] = df_train['Grid_ID'].map(outer_map)
    X = build_design_matrix(df_train, ARM)
    res = pd.read_csv(data.ABLATION_RESULTS)

    rows = []
    for fold in sorted(df_train['outer_fold'].unique()):
        r = res[(res['arm'] == ARM) & (res['outer_fold'] == fold)].iloc[0]
        params, n_est = json.loads(r['best_params']), int(r['n_estimators'])
        te = (df_train['outer_fold'] == fold).to_numpy()
        ytr, yte = df_train.loc[~te, 'Value'].to_numpy(float), df_train.loc[te, 'Value'].to_numpy(float)

        model = xgb.XGBRegressor(n_estimators=n_est, random_state=42, tree_method='hist',
                                 n_jobs=-1, **params)
        model.fit(X[~te], np.log1p(ytr))
        m = metrics(yte, np.expm1(model.predict(X[te])))

        d = df_train.loc[te, ['Location', 'DateTime', 'Value']].sort_values(['Location', 'DateTime'])
        prev = d.groupby('Location')['Value'].shift(1)
        ok = prev.notna().to_numpy()
        p = metrics(yte[ok], prev.to_numpy()[ok])

        print('fold %d: model R2 %.3f (RMSE %.3f) | persistence R2 %.3f (RMSE %.3f) | margin %+.3f'
              % (fold, m['r2'], m['rmse'], p['r2'], p['rmse'], m['r2'] - p['r2']))
        rows.append({'outer_fold': fold, 'test_rows': int(te.sum()),
                     'model_r2': m['r2'], 'model_rmse': m['rmse'],
                     'persistence_r2': p['r2'], 'persistence_rmse': p['rmse']})

    dfm = pd.DataFrame(rows)
    print('\nmean over folds: model R2 %.3f | persistence R2 %.3f | margin %+.3f'
          % (dfm.model_r2.mean(), dfm.persistence_r2.mean(),
             (dfm.model_r2 - dfm.persistence_r2).mean()))
    print('beats persistence in every fold:', bool((dfm.model_r2 > dfm.persistence_r2).all()))
    dfm.to_csv(os.path.join(data.OUT_DIR, 'persistence_by_fold.csv'), index=False)
    print('wrote outputs/persistence_by_fold.csv')


if __name__ == '__main__':
    main()
