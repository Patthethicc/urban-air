import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""Three-way ablation and nested spatial CV on the proper-training cells.

Plan deliverables 5 (Benchmark ablation) and 7 (findings: limits, bottlenecks, strengths).

    python ablation.py --arms A B C
    python ablation.py --arms B --n-trials 5 --smoke     # quick check on one arm

WHAT THIS DOES DIFFERENTLY FROM THE PRE-STEP-3 NOTEBOOK
1. Calibration and test cells are excluded from CV, tuning and fitting. Only the 11 training
   cells in fold_partitions.json are ever seen here, so these numbers are a held-out estimate
   (the old 17-cell numbers were not).
2. Folds come from the stored fold maps, not from a fresh K-Means draw, so all arms are compared
   on identical folds and the run is reproducible.
3. gp_minimize uses acq_func='EI' (plan Section 3.3.3, Eq. 3.28). The sklearn-optimize default is
   'gp_hedge', which the earlier code was silently using.
4. Metrics are reported in raw PM2.5 units after expm1. Model selection inside the inner loop
   stays in log1p units, where the transform is monotonic and the ranking is unchanged.
5. Residuals are actual - predicted (plan Section 3.4.3). The old notebook used predicted - Value,
   which flipped the sign of every residual and of the Moran's I / bias map interpretation.

FEATURE ARMS (plan Section 3.3, cells 7-10: "Define feature groups TIME_ONLY, MORPH_TIME, FULL")
    A  TIME_ONLY   18 features   6 cyclical time + 9 PM2.5 history + 3 interactions
    B  MORPH_TIME  15 features   6 morphology + 6 cyclical time + 3 interactions, NO PM2.5 history
    C  FULL        24 features   morphology + cyclical time + PM2.5 history + interactions
B is the real test of the thesis claim: can urban form predict PM2.5 without being handed the
sensor's own recent history? A is the persistence baseline. C is today's notebook exactly.

OPTIONAL FOURTH ARM (not run by default, plan Section 3.1.5): FULL + the fold-aware spatial lag
(27 features). Deliberately excluded from this pass so the A/B/C comparison stays clean, and
because the lag is a separate deliverable. Run it later with --arms C+ once the core result is on
the record.
"""
import argparse
import json
import os
import time
from math import sqrt

import numpy as np
import pandas as pd
import xgboost as xgb
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import data
from preprocessing import prep
from preprocessing import spatial_cv

# ---------------------------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------------------------
MORPH_FEATURES = ['bh_mean', 'bh_stdev', 'bldg_dens', 'ndvi_mean', 'aspect_ratio_mean',
                  'street_width_mean']
TIME_FEATURES = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin',
                 'month_cos']
PM_HISTORY_FEATURES = ['pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_4', 'pm25_diff_1_2',
                       'pm25_roll24_mean', 'pm25_roll24_std', 'pm25_roll72_mean',
                       'pm25_roll72_std']
INTERACTION_FEATURES = ['hour_x_bldg_dens', 'hour_x_ndvi', 'hour_x_aspect_ratio']
SPATIAL_LAG_FEATURES = ['pm25_spatial_lag', 'pm25_spatial_lag_w', 'pm25_spatial_lag_n']

ARM_FEATURES = {
    'A': TIME_FEATURES + PM_HISTORY_FEATURES + INTERACTION_FEATURES,
    'B': MORPH_FEATURES + TIME_FEATURES + INTERACTION_FEATURES,
    'C': MORPH_FEATURES + TIME_FEATURES + PM_HISTORY_FEATURES + INTERACTION_FEATURES,
    'C+': (MORPH_FEATURES + TIME_FEATURES + PM_HISTORY_FEATURES + INTERACTION_FEATURES
           + SPATIAL_LAG_FEATURES),
}
ARM_LABELS = {
    'A': 'A TIME_ONLY',
    'B': 'B MORPH_TIME',
    'C': 'C FULL',
    'C+': 'C+ FULL+spatial lag',
}

SEARCH_SPACE = [
    Integer(2, 5, name='max_depth'),
    Integer(1, 15, name='min_child_weight'),
    Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree'),
    Real(0.01, 0.3, name='learning_rate', prior='log-uniform'),
    Real(0.0, 5.0, name='gamma'),
    Real(0.1, 10.0, name='reg_lambda', prior='log-uniform'),
    Real(0.0, 10.0, name='reg_alpha'),
]

OUT_DIR = data.OUT_DIR
K_OUTER = 3
K_INNER = 3
N_TRIALS = 20
N_INITIAL_POINTS = 8


def validate_arm_features():
    """Every arm must be a subset of prep.features (plus the lag columns, which are added later),
    and the union of the raw groups must be exactly the import-time feature list."""
    known = set(prep.features) | set(SPATIAL_LAG_FEATURES)
    union = (set(MORPH_FEATURES) | set(TIME_FEATURES) | set(PM_HISTORY_FEATURES)
             | set(INTERACTION_FEATURES))
    assert union == set(prep.features), (
        f'feature groups do not cover prep.features exactly; '
        f'missing={set(prep.features) - union}, extra={union - set(prep.features)}')
    for arm, cols in ARM_FEATURES.items():
        unknown = set(cols) - known
        assert not unknown, f'arm {arm} references unknown columns: {unknown}'
    assert ARM_FEATURES['C'] == list(prep.features), 'arm C must be the import-time 24 columns'


def build_design_matrix(df_clean, arm, spatial_lag=None):
    """X for one arm. If `spatial_lag` is given (a frame from compute_spatial_lag), the lag columns
    are merged on (Grid_ID, DateTime) before selecting columns, so C+ sees fold-aware lag values."""
    data = df_clean
    if spatial_lag is not None:
        data = df_clean.merge(spatial_lag, on=['Grid_ID', 'DateTime'], how='left')
    return data[ARM_FEATURES[arm]].values


def tune_and_evaluate(X_arm, y, df_train, outer_fold, inner_map, n_trials, random_state=42,
                      verbose=True, n_initial_points=N_INITIAL_POINTS):
    """Nested CV for ONE outer fold: Bayesian search on inner folds, then score the held-out fold.

    Returns (metrics dict, out-of-fold frame for this fold).
    """
    held_out_cells = sorted(df_train.loc[df_train['outer_fold'] == outer_fold, 'Grid_ID'].unique())
    train_mask = (df_train['outer_fold'] != outer_fold).values
    test_mask = (df_train['outer_fold'] == outer_fold).values

    inner_labels = df_train.loc[train_mask, 'Grid_ID'].map(inner_map).values
    if pd.isna(inner_labels).any():
        raise ValueError(f'inner fold map {inner_map} does not cover every training cell of '
                         f'outer fold {outer_fold}')
    inner_labels = inner_labels.astype(int)

    X_tr, y_tr = X_arm[train_mask], y[train_mask]
    X_te, y_te = X_arm[test_mask], y[test_mask]

    call_records = []

    @use_named_args(SEARCH_SPACE)
    def objective(**params):
        inner_rmses, best_iters = [], []
        for inner_f in sorted(set(inner_labels)):
            itr = inner_labels != inner_f
            ite = inner_labels == inner_f
            model = xgb.XGBRegressor(n_estimators=500, early_stopping_rounds=20,
                                     random_state=random_state, tree_method='hist', n_jobs=-1,
                                     **params)
            model.fit(X_tr[itr], y_tr[itr], eval_set=[(X_tr[ite], y_tr[ite])], verbose=False)
            preds = model.predict(X_tr[ite])
            # inner-loop selection stays in log1p units (monotonic transform, same ranking)
            inner_rmses.append(sqrt(mean_squared_error(y_tr[ite], preds)))
            best_iters.append(model.best_iteration)
        avg_rmse = float(np.mean(inner_rmses))
        call_records.append({'params': dict(params), 'avg_rmse': avg_rmse,
                             'avg_best_iteration': float(np.mean(best_iters))})
        return avg_rmse

    gp_minimize(objective, SEARCH_SPACE, n_calls=n_trials, n_initial_points=n_initial_points,
                acq_func='EI', random_state=random_state, verbose=False)

    best_call = min(call_records, key=lambda r: r['avg_rmse'])
    best_params = {k: (int(v) if isinstance(v, np.integer) else float(v))
                   for k, v in best_call['params'].items()}
    best_n_estimators = max(10, int(round(best_call['avg_best_iteration'])))

    final_model = xgb.XGBRegressor(n_estimators=best_n_estimators, random_state=random_state,
                                   tree_method='hist', n_jobs=-1, **best_params)
    final_model.fit(X_tr, y_tr)

    preds_raw = np.expm1(final_model.predict(X_te))
    actual_raw = np.expm1(y_te)
    metrics = {
        'outer_fold': int(outer_fold),
        'n_test_rows': int(test_mask.sum()),
        'n_train_rows': int(train_mask.sum()),
        'n_test_cells': len(held_out_cells),
        'test_cells': ';'.join(str(c) for c in held_out_cells),
        'inner_rmse_log1p': round(float(best_call['avg_rmse']), 4),
        'rmse': float(sqrt(mean_squared_error(actual_raw, preds_raw))),
        'mae': float(mean_absolute_error(actual_raw, preds_raw)),
        'r2': float(r2_score(actual_raw, preds_raw)),
        'n_estimators': best_n_estimators,
        'best_params': json.dumps(best_params, sort_keys=True),
    }
    oof = df_train.loc[test_mask, ['Location', 'Grid_ID', 'DateTime', 'Value']].copy()
    oof['actual'] = actual_raw
    oof['predicted'] = preds_raw
    oof['residual'] = oof['actual'] - oof['predicted']   # actual - predicted, plan Section 3.4.3
    oof['outer_fold'] = int(outer_fold)

    if verbose:
        print(f"    outer fold {outer_fold}: cells={held_out_cells} n={metrics['n_test_rows']} "
              f"RMSE={metrics['rmse']:.3f} MAE={metrics['mae']:.3f} R2={metrics['r2']:.3f} "
              f"| n_est={best_n_estimators}")
    return metrics, oof


def run(arms=('A', 'B', 'C'), n_trials=N_TRIALS, smoke=False, out_dir=OUT_DIR, seed=42):
    validate_arm_features()
    df = prep.df_clean.copy()
    y_all = np.log1p(df['Value'].values)

    doc = spatial_cv.load_fold_map(data.FOLD_JSON)
    parts = doc['partitions']
    train_cells = list(parts['train'])
    outer_map = {int(c): int(f) for f, cells in doc['folds']['outer'].items() for c in cells}
    inner_maps = {int(f): {int(c): int(v) for c, v in m.items()}
                  for f, m in doc['folds']['inner'].items()}

    # Only the proper-training cells enter CV, tuning or fitting.
    df_train = df[df['Grid_ID'].isin(train_cells)].reset_index(drop=True)
    y = np.log1p(df_train['Value'].values)
    df_train['outer_fold'] = df_train['Grid_ID'].map(outer_map)
    assert df_train['outer_fold'].notna().all(), 'a training cell is missing from the outer fold map'
    assert not set(df_train['Grid_ID']) & (set(parts['calibration']) | set(parts['test']))

    os.makedirs(out_dir, exist_ok=True)
    all_results, all_oof = [], {}

    for arm in arms:
        if arm == 'C+':
            raise SystemExit('C+ needs a fold-aware spatial lag computed per outer fold; not '
                             'implemented in this pass on purpose (see module docstring).')
        t0 = time.time()
        print(f"\n=== arm {ARM_LABELS[arm]} ({len(ARM_FEATURES[arm])} features) ===")
        X_arm = build_design_matrix(df_train, arm)
        folds = sorted(df_train['outer_fold'].unique())

        for fold in (folds[:1] if smoke else folds):
            metrics, oof = tune_and_evaluate(
                X_arm, y, df_train, fold, inner_maps[int(fold)],
                n_trials=(3 if smoke else n_trials), random_state=seed,
                n_initial_points=(2 if smoke else N_INITIAL_POINTS))
            metrics['arm'] = arm
            metrics['n_features'] = len(ARM_FEATURES[arm])
            all_results.append(metrics)
            all_oof.setdefault(arm, []).append(oof)

            # checkpoint after every fold so a crash does not lose the Bayesian search
            pd.DataFrame(all_results).to_csv(os.path.join(out_dir, 'ablation_results.csv'),
                                             index=False)
            pd.concat(all_oof[arm], ignore_index=True).to_csv(
                os.path.join(out_dir, f'oof_predictions_{arm.replace("+", "plus")}.csv'),
                index=False)

        res = pd.DataFrame(all_results)
        res = res[res['arm'] == arm]
        print(f"  arm {arm} done in {time.time() - t0:.1f}s | "
              f"mean RMSE {res['rmse'].mean():.3f} MAE {res['mae'].mean():.3f} "
              f"R2 {res['r2'].mean():.3f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(out_dir, 'ablation_results.csv'), index=False)

    summary_rows = []
    for arm, g in results_df.groupby('arm', sort=False):
        oof = pd.concat(all_oof[arm], ignore_index=True)
        summary_rows.append({
            'arm': arm,
            'label': ARM_LABELS[arm],
            'n_features': int(g['n_features'].iloc[0]),
            'n_outer_folds': len(g),
            'rmse_mean': g['rmse'].mean(), 'rmse_sd': g['rmse'].std(),
            'mae_mean': g['mae'].mean(), 'mae_sd': g['mae'].std(),
            'r2_mean': g['r2'].mean(), 'r2_sd': g['r2'].std(),
            'pooled_rmse': sqrt(mean_squared_error(oof['actual'], oof['predicted'])),
            'pooled_mae': mean_absolute_error(oof['actual'], oof['predicted']),
            'pooled_r2': r2_score(oof['actual'], oof['predicted']),
            'oof_rows': len(oof),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, 'ablation_summary.csv'), index=False)

    print('\n=== ablation summary ===')
    print(summary.round(3).to_string(index=False))

    # the three arms must be scored on exactly the same rows, or the comparison is confounded
    if len(all_oof) > 1:
        frames = {arm: pd.concat(chunks, ignore_index=True) for arm, chunks in all_oof.items()}
        keys = {arm: set(map(tuple, d[['Grid_ID', 'DateTime']].values))
                for arm, d in frames.items()}
        first = next(iter(keys))
        for arm in keys:
            assert keys[arm] == keys[first], f'arm {arm} was scored on different rows'
        print(f'\nall arms scored on identical rows: {len(keys[first])}')

    with open(os.path.join(out_dir, 'ablation_params.json'), 'w', encoding='utf-8') as fh:
        json.dump({'arms': {a: ARM_FEATURES[a] for a in all_oof},
                   'k_outer': K_OUTER, 'k_inner': K_INNER, 'n_trials': n_trials,
                   'acq_func': 'EI', 'residual': 'actual - predicted',
                   'train_cells': train_cells,
                   'calibration_cells': parts['calibration'], 'test_cells': parts['test']},
                  fh, indent=2, sort_keys=True)
    return results_df, summary, all_oof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arms', nargs='+', default=['A', 'B', 'C'], choices=['A', 'B', 'C', 'C+'])
    ap.add_argument('--n-trials', type=int, default=N_TRIALS)
    ap.add_argument('--smoke', action='store_true', help='one outer fold, 3 trials')
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    run(arms=args.arms, n_trials=args.n_trials, smoke=args.smoke, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
