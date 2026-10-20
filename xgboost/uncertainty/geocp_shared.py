"""Phase 4: a GeoCP configuration that matches the CNN-LSTM run, so coverage numbers are comparable.

Their `configs/geocp.yaml` (recorded in `data/20261016_070432/geocp.json`) uses:

    alpha 0.1 · kernels gaussian AND exponential · bandwidths [2000, 5000, 10000, 20000, inf]
    finite_sample_correction False · min_lower_bound 0
    selection: leave-one-station-out on the CALIBRATION stations, pick the config that meets the
               coverage target and then has the smallest mean width

Ours used a finer bandwidth grid starting at 500 m, a single kernel, and the finite-sample correction
on. Any coverage comparison across branches is meaningless unless those match, so this reproduces their
search and records the full candidate table the way their geocp.json does.

    python -m uncertainty.geocp_shared                 # writes outputs/geocp_shared/
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import data
from uncertainty import geocp

# their grid, their kernels, their correction setting
SHARED_H_GRID = np.array([2000.0, 5000.0, 10000.0, 20000.0, np.inf])
SHARED_KERNELS = ('gaussian', 'exponential')
SHARED_ALPHA = 0.10
SHARED_CORRECTION = False
SHARED_TOLERANCE = 0.02

OUT = os.path.join(data.OUT_DIR, 'geocp_shared')


def run(max_cal=6000, max_eval=20000, seed=42, out_dir=OUT):
    os.makedirs(out_dir, exist_ok=True)
    ctx = geocp.load_context()

    n_test_all = len(ctx['test'])
    if max_eval and n_test_all > max_eval:
        ctx['test'] = (ctx['test'].sample(max_eval, random_state=seed)
                       .sort_values(['Grid_ID', 'cx', 'cy']).reset_index(drop=True))
        print('test rows subsampled for evaluation: %s of %s (seed %d)'
              % (f'{max_eval:,}', f'{n_test_all:,}', seed))

    cal = geocp.calibration_scores(ctx)
    n_cal_all = len(cal)
    if max_cal and n_cal_all > max_cal:
        cal = (cal.sample(max_cal, random_state=seed)
               .sort_values(['Grid_ID', 'cx', 'cy']).reset_index(drop=True))
        print('calibration records subsampled: %s of %s (seed %d)'
              % (f'{max_cal:,}', f'{n_cal_all:,}', seed))

    query_xy = ctx['test'][['cx', 'cy']].values
    # `load_context` gives raw feature rows; the point predictions come from applying the production
    # model. (An earlier version read ctx['test']['predicted'], which does not exist -> KeyError.)
    test_pred = geocp.predict_raw(ctx['model'], ctx['test'])
    print('\n=== leave-one-calibration-station-out over kernels x bandwidths ===')

    candidates = []
    for kernel in SHARED_KERNELS:
        for h in SHARED_H_GRID:
            per_fold_cov, per_fold_w = [], []
            for held in sorted(cal['Grid_ID'].unique()):
                fit = cal[cal['Grid_ID'] != held]
                sel = cal[cal['Grid_ID'] == held]
                if fit.empty or sel.empty:
                    continue
                res = geocp.intervals_for_cells(fit, sel[['cx', 'cy']].values,
                                                sel['predicted'].values, SHARED_ALPHA,
                                                h, kernel, SHARED_CORRECTION)
                obs = sel['observed'].values
                per_fold_cov.append(float(((obs >= res['lower']) & (obs <= res['upper'])).mean()))
                per_fold_w.append(float(res['q'].mean() * 2.0))
            if not per_fold_cov:
                continue
            candidates.append({'kernel': kernel, 'bandwidth_m': None if np.isinf(h) else float(h),
                               'loso_coverage': float(np.mean(per_fold_cov)),
                               'loso_worst_fold_coverage': float(np.min(per_fold_cov)),
                               'loso_mean_width': float(np.mean(per_fold_w))})
        print('  %-12s done' % kernel)

    cand = pd.DataFrame(candidates).sort_values(['loso_worst_fold_coverage', 'loso_mean_width'],
                                                ascending=[False, True])
    cand.to_csv(os.path.join(out_dir, 'geocp_candidates.csv'), index=False)

    target = 1.0 - SHARED_ALPHA
    ok = cand[cand['loso_worst_fold_coverage'] >= target - SHARED_TOLERANCE]
    best = (ok.sort_values('loso_mean_width').iloc[0] if len(ok) else cand.iloc[0])
    met = bool(best['loso_worst_fold_coverage'] >= target - SHARED_TOLERANCE)
    print('\nselected: kernel=%s  h=%s' % (best['kernel'], best['bandwidth_m']))
    print('  LOSO mean coverage %.4f | WORST fold %.4f | mean width %.3f'
          % (best['loso_coverage'], best['loso_worst_fold_coverage'], best['loso_mean_width']))
    print('  rule: worst fold must be >= %.2f (target %.2f - tolerance %.2f) -> %s'
          % (target - SHARED_TOLERANCE, target, SHARED_TOLERANCE,
             'MET' if met else 'NOT MET (report as an under-covering pilot)'))

    h = np.inf if pd.isna(best['bandwidth_m']) else float(best['bandwidth_m'])
    test_iv = geocp.intervals_for_cells(cal, query_xy, test_pred,
                                        SHARED_ALPHA, h, best['kernel'], SHARED_CORRECTION)
    # ctx['test'] carries the target as `Value`; only calibration_scores() renames it to
    # `observed`. Using ['observed'] here raised KeyError.
    obs = ctx['test']['Value'].to_numpy(float)
    cov = float(((obs >= test_iv['lower']) & (obs <= test_iv['upper'])).mean())
    width = float(test_iv['q'].mean() * 2.0)
    print('  test coverage %.4f (target %.2f), mean width %.3f' % (cov, target, width))

    split = geocp.intervals_for_cells(cal, query_xy, test_pred,
                                      SHARED_ALPHA, np.inf, 'uniform', True)
    split_cov = float(((obs >= split['lower']) & (obs <= split['upper'])).mean())

    summary = {
        'selected': {'kernel': best['kernel'],
                     'bandwidth_m': None if np.isinf(h) else float(h),
                     'alpha': SHARED_ALPHA, 'finite_sample_correction': SHARED_CORRECTION,
                     'loso_coverage': float(best['loso_coverage']),
                     'loso_mean_width': float(best['loso_mean_width']),
                     'met_coverage_target': met},
        'test': {'target_coverage': target, 'coverage': cov, 'mean_width': width,
                 'mean_half_width': width / 2.0,
                 'rows': int(len(obs)),
                 'unweighted_split_conformal': {'coverage': split_cov,
                                                'mean_width': float(split['q'].mean() * 2.0)}},
        'inputs': {'calibration_records': int(len(cal)),
                   'calibration_records_available': int(n_cal_all),
                   'test_rows': int(len(obs)), 'test_rows_available': int(n_test_all),
                   'kernels': list(SHARED_KERNELS),
                   'bandwidths_m': [None if np.isinf(v) else float(v) for v in SHARED_H_GRID],
                   'seed': seed},
        'comparable_to': 'CNN-LSTM run data/20261016_070432 (test coverage 0.906, width 15.39)',
    }
    with open(os.path.join(out_dir, 'geocp.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, default=str)

    iv = ctx['test'][['Location', 'Grid_ID', 'DateTime']].copy()
    iv['observed'] = obs
    iv['prediction'] = test_pred
    iv['geocp_lower'] = test_iv['lower'].values
    iv['geocp_upper'] = test_iv['upper'].values
    iv['geocp_width'] = test_iv['width'].values
    iv['geocp_covered'] = ((obs >= test_iv['lower']) & (obs <= test_iv['upper'])).values
    iv.to_csv(os.path.join(out_dir, 'predictions_test_intervals.csv'), index=False)
    print('\nwrote %s/ (geocp.json, geocp_candidates.csv, predictions_test_intervals.csv)' % out_dir)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-cal', type=int, default=6000)
    ap.add_argument('--max-eval', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default=OUT)
    a = ap.parse_args()
    run(a.max_cal, a.max_eval, a.seed, a.out_dir)


if __name__ == '__main__':
    main()
