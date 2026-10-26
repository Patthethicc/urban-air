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

# ---- 10 m patch statistics (preprocessing/patch_stats.py) --------------------------------------
# The CNN-LSTM branch reads a 32x32 px patch at 10 m (320 m) and convolves it, so it sees how
# morphology is ARRANGED inside a cell. Six zonal means per 500 m cell cannot express that, which made
# any "CNN beats XGBoost" claim a comparison of inputs rather than architectures. These 35 columns are
# the same 320 m window reduced to summary statistics, so the tabular model can compete on texture.
TEXTURE_BANDS = ['bh', 'bh_std_px', 'ndvi', 'dens', 'aspect']
TEXTURE_STATS = ['mean', 'std', 'min', 'max', 'p10', 'p50', 'p90']
TEXTURE_FEATURES = ['tex_%s_%s' % (b, st) for b in TEXTURE_BANDS for st in TEXTURE_STATS]
SPATIAL_LAG_FEATURES = ['pm25_spatial_lag', 'pm25_spatial_lag_w', 'pm25_spatial_lag_n']

ARM_FEATURES = {
    'A': TIME_FEATURES + PM_HISTORY_FEATURES + INTERACTION_FEATURES,
    'B': MORPH_FEATURES + TIME_FEATURES + INTERACTION_FEATURES,
    'C': MORPH_FEATURES + TIME_FEATURES + PM_HISTORY_FEATURES + INTERACTION_FEATURES,
    'TEXTURE': (MORPH_FEATURES + TIME_FEATURES + PM_HISTORY_FEATURES
                + INTERACTION_FEATURES + TEXTURE_FEATURES),
    'C+': (MORPH_FEATURES + TIME_FEATURES + PM_HISTORY_FEATURES + INTERACTION_FEATURES
           + SPATIAL_LAG_FEATURES),
}
ARM_LABELS = {
    'A': 'A TIME_ONLY',
    'B': 'B MORPH_TIME',
    'C': 'C FULL',
    'TEXTURE': 'D TEXTURE',
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

# The untuned control: XGBoost's own defaults, used to measure what the Bayesian search bought.
# Deliberately conservative and unremarkable -- the point is that no search touched them.
DEFAULT_PARAMS = {'colsample_bytree': 0.8, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 6,
                  'min_child_weight': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 1.0, 'subsample': 1.0}


def validate_arm_features():
    """Every arm must be a subset of prep.features (plus the lag columns, which are added later),
    and the union of the raw groups must be exactly the import-time feature list."""
    known = set(prep.features) | set(SPATIAL_LAG_FEATURES) | set(TEXTURE_FEATURES)
    union = (set(MORPH_FEATURES) | set(TIME_FEATURES) | set(PM_HISTORY_FEATURES)
             | set(INTERACTION_FEATURES))
    assert union == set(prep.features), (
        f'feature groups do not cover prep.features exactly; '
        f'missing={set(prep.features) - union}, extra={union - set(prep.features)}')
    for arm, cols in ARM_FEATURES.items():
        unknown = set(cols) - known
        assert not unknown, f'arm {arm} references unknown columns: {unknown}'
    assert ARM_FEATURES['C'] == list(prep.features), 'arm C must be the import-time 24 columns'


_TEXTURE_CACHE = {}


def texture_table():
    """data/patch_stats.csv, cached. One row per sensor cell, keyed by Grid_ID."""
    if 'df' not in _TEXTURE_CACHE:
        import pandas as pd
        path = os.path.join(data.DATA_DIR, 'patch_stats.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} is missing: run `python -m preprocessing.patch_stats` first '
                f'(it builds the 10 m patch statistics the TEXTURE arm needs).')
        _TEXTURE_CACHE['df'] = pd.read_csv(path)
    return _TEXTURE_CACHE['df']


def build_design_matrix(df_clean, arm, spatial_lag=None):
    """X for one arm. If `spatial_lag` is given (a frame from compute_spatial_lag), the lag columns
    are merged on (Grid_ID, DateTime) before selecting columns, so C+ sees fold-aware lag values.
    TEXTURE additionally merges the per-cell 10 m patch statistics on Grid_ID."""
    data = df_clean
    if spatial_lag is not None:
        data = data.merge(spatial_lag, on=['Grid_ID', 'DateTime'], how='left')
    if arm == 'TEXTURE':
        tex = texture_table()
        data = data.merge(tex[['Grid_ID'] + TEXTURE_FEATURES], on='Grid_ID', how='left')
    return data[ARM_FEATURES[arm]].values


def tune_and_evaluate(X_arm, y, df_train, outer_fold, inner_map, n_trials, random_state=42,
                      verbose=True, n_initial_points=N_INITIAL_POINTS, arm=None, k_lag=None,
                      row_stride=1, params_fixed=None):
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

    # C+ needs a lag built from THIS fold's fit cells only, so the design matrix cannot be shared
    # across folds: the neighbours a row may see exclude both its own cell and the held-out cells.
    if k_lag is not None and arm is not None and 'spatial' in ARM_LABELS.get(arm, '').lower():
        held = sorted(int(c) for c in df_train.loc[test_mask, 'Grid_ID'].unique())
        fit_cells = [int(c) for c in df_train['Grid_ID'].unique() if int(c) not in held]
        print('    fold %d: computing fold-aware spatial lag (k=%d, %d fit cells, %d held out) ...'
              % (outer_fold, k_lag, len(fit_cells), len(held)), flush=True)
        lag = prep.compute_spatial_lag(
            df_train, allowed_grid_ids=fit_cells,
            target_grid_ids=sorted(int(c) for c in df_train['Grid_ID'].unique()), k=k_lag)
        X_arm = build_design_matrix(df_train, arm, spatial_lag=lag)
        present = float(lag['pm25_spatial_lag'].notna().mean())
        print('      lag present on %.1f%% of cell-hours' % (100.0 * present), flush=True)

    X_tr, y_tr = X_arm[train_mask], y[train_mask]
    X_te, y_te = X_arm[test_mask], y[test_mask]

    call_records = []

    # Tuning on every row is what makes a 20-trial search cost 2.5 h. Striding the TRAINING rows of
    # each inner fit keeps every validation row (so early stopping still sees the real signal) while
    # making each trial proportionally cheaper -- the search can then afford 60 trials instead of 20.
    _stride = max(1, int(row_stride))

    @use_named_args(SEARCH_SPACE)
    def objective(**params):
        # skopt hands integer dimensions back as 6.0, which XGBoost rejects for max_depth
        params = {k: (int(round(v)) if k == 'max_depth' else v) for k, v in params.items()}
        inner_rmses, inner_r2s, best_iters = [], [], []
        for inner_f in sorted(set(inner_labels)):
            itr = inner_labels != inner_f
            ite = inner_labels == inner_f
            if _stride > 1:
                idx = np.flatnonzero(itr)
                keep = np.zeros_like(itr)
                keep[idx[::_stride]] = True
                itr = keep
            model = xgb.XGBRegressor(n_estimators=500, early_stopping_rounds=20,
                                     random_state=random_state, tree_method='hist', n_jobs=-1,
                                     **params)
            model.fit(X_tr[itr], y_tr[itr], eval_set=[(X_tr[ite], y_tr[ite])], verbose=False)
            preds = model.predict(X_tr[ite])
            # inner-loop selection stays in log1p units (monotonic transform, same ranking)
            inner_rmses.append(sqrt(mean_squared_error(y_tr[ite], preds)))
            # R2 on the ug/m3 scale (not log1p): this is the figure the results tables quote, and it is
            # the number a reader can compare with the outer-fold R2 printed a few lines later.
            _act, _pred = np.expm1(y_tr[ite]), np.expm1(preds)
            _ss_tot = float((( _act - _act.mean()) ** 2).sum())
            inner_r2s.append(1.0 - float(((_act - _pred) ** 2).sum()) / _ss_tot if _ss_tot else np.nan)
            best_iters.append(model.best_iteration)
        avg_rmse = float(np.mean(inner_rmses))
        avg_r2 = float(np.nanmean(inner_r2s))
        call_records.append({'params': dict(params), 'avg_rmse': avg_rmse, 'avg_r2': avg_r2,
                             'avg_best_iteration': float(np.mean(best_iters))})
        return avg_rmse

    # ---- progress reporting -------------------------------------------------------------
    # A fold is 20 Bayesian trials with no other output, which is minutes of silence per fold and
    # hours across a run. gp_minimize calls `callback` after every iteration, so this prints one
    # line per trial (trial number, that trial's inner RMSE, best so far, elapsed). flush=True so
    # the line appears immediately even when stdout is redirected to a file.
    t_fold = time.time()
    n_inner = len(set(inner_labels))

    def _progress(res):
        done = len(res.func_vals)
        best = float(np.min(res.func_vals))
        cur = float(res.func_vals[-1])
        elapsed = time.time() - t_fold
        per_trial = elapsed / max(1, done)
        eta = per_trial * max(0, n_trials - done)
        mark = '  <- best' if cur <= best else ''
        cur_r2 = call_records[-1]['avg_r2'] if call_records else float('nan')
        best_i = int(np.argmin(res.func_vals))
        best_r2 = call_records[best_i]['avg_r2'] if best_i < len(call_records) else float('nan')
        print(f"      trial {done:2d}/{n_trials}  inner_rmse {cur:.5f}  inner_r2 {cur_r2:.4f}  "
              f"best {best:.5f} (r2 {best_r2:.4f}){mark}  {elapsed:5.0f}s  eta {eta:4.0f}s",
              flush=True)

    if params_fixed is not None:
        # Untuned control: same fold, same striding, no search -- one fixed parameter set.
        _keys = [d.name for d in SEARCH_SPACE]
        _missing = [k for k in _keys if k not in params_fixed]
        assert not _missing, f'params_fixed is missing {_missing}'
        if verbose:
            print(f"    fold {outer_fold}: FIXED parameters, no search "
                  f"({len(X_tr):,} train rows, stride {_stride}) ...", flush=True)
        # use_named_args exposes the decorated callable positionally, in SEARCH_SPACE order
        objective([params_fixed[d.name] for d in SEARCH_SPACE])
    else:
        if verbose:
            print(f"    fold {outer_fold}: tuning ({n_trials} trials x {n_inner} inner folds, "
                  f"{len(X_tr):,} train rows, stride {_stride}) ...", flush=True)
        # skopt insists on n_calls > n_initial_points. Keep the thesis setting (8) but shrink the
        # random phase for short searches, so `--n-trials 3` gives a quick read instead of a crash.
        n_initial_points = max(2, min(n_initial_points, n_trials - 1))
        gp_minimize(objective, SEARCH_SPACE, n_calls=n_trials, n_initial_points=n_initial_points,
                    acq_func='EI', random_state=random_state, verbose=False,
                    callback=_progress if verbose else None)

    best_call = min(call_records, key=lambda r: r['avg_rmse'])
    # max_depth must stay an int: XGBoost rejects '6.0'. Everything else may be float.
    best_params = {k: (int(round(v)) if k == 'max_depth' else float(v))
                   for k, v in best_call['params'].items()}
    best_n_estimators = max(10, int(round(best_call['avg_best_iteration'])))

    if verbose:
        print(f"    fold {outer_fold}: {'FIXED' if params_fixed else 'search done'}, "
              f"inner_rmse {best_call['avg_rmse']:.5f} "
              f"inner_r2 {best_call.get('avg_r2', float('nan')):.4f} "
              f"-> n_estimators={best_n_estimators}; fitting final model on {len(X_tr):,} rows ...",
              flush=True)

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
        'inner_r2': round(float(best_call.get('avg_r2', float('nan'))), 4),
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
              f"| n_est={best_n_estimators} | fold took {time.time() - t_fold:.0f}s", flush=True)
    return metrics, oof


def run(arms=('A', 'B', 'C'), n_trials=N_TRIALS, smoke=False, out_dir=OUT_DIR, seed=42,
        row_stride=1, untuned=False):
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
        k_lag = 5 if arm == 'C+' else None      # 5 nearest stations, matching the CNN-LSTM branch
        t0 = time.time()
        print(f"\n=== arm {ARM_LABELS[arm]} ({len(ARM_FEATURES[arm])} features) "
              f"[arm {arms.index(arm) + 1}/{len(arms)}] ===", flush=True)
        # C+ builds its design matrix per fold (the lag is fold-dependent); every other arm shares one.
        X_arm = None if arm == 'C+' else build_design_matrix(df_train, arm)
        folds = sorted(df_train['outer_fold'].unique())

        print(f"    {len(folds)} outer fold(s): {folds}", flush=True)
        for fold in (folds[:1] if smoke else folds):
            metrics, oof = tune_and_evaluate(
                X_arm, y, df_train, fold, inner_maps[int(fold)],
                n_trials=(3 if smoke else n_trials), random_state=seed,
                n_initial_points=(2 if smoke else N_INITIAL_POINTS),
                arm=arm, k_lag=k_lag, row_stride=row_stride,
                params_fixed=(DEFAULT_PARAMS if untuned else None))
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
    ap.add_argument('--arms', nargs='+', default=['A', 'B', 'C'],
                    choices=['A', 'B', 'C', 'C+', 'TEXTURE'])
    ap.add_argument('--n-trials', type=int, default=N_TRIALS)
    ap.add_argument('--smoke', action='store_true', help='one outer fold, 3 trials')
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    run(arms=args.arms, n_trials=args.n_trials, smoke=args.smoke, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
