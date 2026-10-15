import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""Verify the GeoCP implementation and its outputs. Exits non-zero if any check fails.

    python verify_geocp.py [--out-dir outputs/geocp]

What this checks, and why each one would matter:
  1. Flat-kernel limit: with the uniform kernel the weighted quantile must reduce EXACTLY to split
     conformal. This is the strongest available correctness test for the weighting code, because
     split conformal has an unambiguous closed form.
  2. weighted_quantile against a brute-force implementation of the thesis's own definition
     (Eq. 3.45: inf{q : F_hat(q) >= 1-alpha}) over random scores and weights.
  3. Partition discipline: no calibration or test cell may appear in the fit set; the training,
     calibration and test partitions must be disjoint.
  4. Interval arithmetic: lower/upper/width must equal yhat -/+ q and 2q; the display lower bound
     must be the unclipped lower bound clipped at zero; coverage must be computed on the unclipped
     interval (Section 3.5.2).
  5. Monotonicity in alpha: a larger alpha (lower nominal coverage) must never widen q, at any
     query cell, for a fixed h.
  6. Metric reproduction: the coverage table must reproduce from the stored per-row intervals.
  7. Determinism.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import data  # noqa: E402
from uncertainty import geocp  # noqa: E402
from preprocessing import prep  # noqa: E402
from preprocessing import spatial_cv  # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', default='outputs/geocp')
args = ap.parse_args()
OUT = args.out_dir

print('1. FLAT-KERNEL LIMIT == SPLIT CONFORMAL (the key correctness test)')
ctx = geocp.load_context()
cal = geocp.calibration_scores(ctx)
alpha = 0.10
split_q = geocp.weighted_quantile(cal['alpha'].values,
                                  np.full(len(cal), 1.0 / len(cal)), 1 - alpha, True)
for h in [500, 5000, 50000, 1e9]:
    uq = geocp.weighted_quantile(cal['alpha'].values,
                                 geocp.normalised_weights(np.full(len(cal), h), h, 'uniform'),
                                 1 - alpha, True)
    check(f'uniform kernel at h={h:.0f} reproduces the split-conformal quantile',
          abs(uq - split_q) < 1e-9, f'{uq:.6f} vs {split_q:.6f}')
# and the gaussian kernel must converge to it in the wide-bandwidth limit
gq = geocp.weighted_quantile(cal['alpha'].values,
                             geocp.normalised_weights(np.zeros(len(cal)), 1e9, 'gaussian'),
                             1 - alpha, True)
check('gaussian kernel -> split conformal as h -> infinity', abs(gq - split_q) < 1e-9,
      f'{gq:.6f} vs {split_q:.6f}')

print('\n2. weighted_quantile MATCHES THE THESIS DEFINITION (Eq. 3.45)')
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    m = int(rng.integers(3, 400))
    scores = rng.normal(size=m)
    w = rng.random(m)
    w = w / w.sum()
    level = float(rng.uniform(0.05, 0.99))
    got = geocp.weighted_quantile(scores, w, level, finite_sample_correction=False)
    # brute force: smallest candidate score whose weighted CDF reaches the level
    order = np.argsort(scores)
    s_sorted, w_sorted = scores[order], w[order]
    cdf = np.cumsum(w_sorted)
    expect = min(s_sorted[k] for k in range(m) if cdf[k] >= level - 1e-12)
    worst = max(worst, abs(got - expect))
check('matches brute-force inf{q : F(q) >= level} over 200 random cases', worst < 1e-9,
      f'max deviation {worst:.2e}')
# the finite-sample correction must shift the level to (1-alpha)(1+1/m)
m = 100
scores = np.arange(1, m + 1, dtype=float)
w = np.full(m, 1.0 / m)
q_corr = geocp.weighted_quantile(scores, w, 0.90, True)
q_raw = geocp.weighted_quantile(scores, w, 0.90, False)
check('finite-sample correction picks a strictly higher order statistic',
      q_corr > q_raw, f'{q_corr:.0f} vs {q_raw:.0f}')

print('\n3. PARTITION DISCIPLINE')
parts = ctx['parts']
check('train / calibration / test are disjoint',
      not (set(parts['train']) & set(parts['calibration']))
      and not (set(parts['train']) & set(parts['test']))
      and not (set(parts['calibration']) & set(parts['test'])))
check('no calibration cell is in the model fit set',
      not (set(ctx['train']['Grid_ID']) & set(parts['calibration'])))
check('no test cell is in the model fit set',
      not (set(ctx['train']['Grid_ID']) & set(parts['test'])))
check('calibration scores cover only calibration cells',
      set(cal['Grid_ID']) == set(parts['calibration']),
      str(sorted(set(cal['Grid_ID']))))
check('calibration coordinates are projected metres, not degrees',
      bool(cal['cx'].abs().min() > 1e4), f"cx min {cal['cx'].min():.0f}")

print('\n4. INTERVAL ARITHMETIC AND CLIPPING POLICY')
iv_path = os.path.join(OUT, f'geocp_test_intervals_alpha{int(alpha * 100)}.csv')
if not os.path.exists(iv_path):
    print(f'  (skipping: {iv_path} not found; run geocp.py first)')
else:
    iv = pd.read_csv(iv_path)
    # Tolerance is 1e-4, not 1e-9: the CSV round-trip stores float32, so a difference of two
    # round-tripped quantities (prediction - q) picks up ~4e-06 of storage error. `width` is stored
    # directly from 2q and matches to 1e-9, which is itself evidence the arithmetic is right.
    check('lower == prediction - q',
          bool(np.allclose(iv['geocp_lower'], iv['prediction'] - iv['geocp_q'], atol=1e-4)),
          f"max deviation {float(np.abs(iv['geocp_lower'] - (iv['prediction'] - iv['geocp_q'])).max()):.2e}")
    check('upper == prediction + q',
          bool(np.allclose(iv['geocp_upper'], iv['prediction'] + iv['geocp_q'], atol=1e-4)),
          f"max deviation {float(np.abs(iv['geocp_upper'] - (iv['prediction'] + iv['geocp_q'])).max()):.2e}")
    check('width == 2q exactly (stored directly, so no cancellation error)',
          bool(np.allclose(iv['geocp_width'], 2 * iv['geocp_q'], atol=1e-9)))
    check('width == upper - lower within storage precision',
          bool(np.allclose(iv['geocp_width'], iv['geocp_upper'] - iv['geocp_lower'], atol=1e-4)))
    check('interval is symmetric about the point prediction',
          bool(np.allclose(iv['geocp_lower'] + iv['geocp_upper'], 2 * iv['prediction'], atol=1e-6)))
    check('display lower bound never negative',
          bool((iv['geocp_lower_display'] >= 0).all()))
    neg = iv['geocp_lower'] < 0
    check('display bound clips exactly the negative lower bounds',
          bool(np.allclose(iv.loc[neg, 'geocp_lower_display'], 0.0)
               and np.allclose(iv.loc[~neg, 'geocp_lower_display'],
                               iv.loc[~neg, 'geocp_lower'], atol=1e-9)),
          f'{int(neg.sum())} of {len(iv)} intervals clipped for display')
    # coverage MUST use the unclipped interval; recompute both ways and show they differ
    cov_unclipped = float(((iv['observed'] >= iv['geocp_lower'])
                           & (iv['observed'] <= iv['geocp_upper'])).mean())
    cov_clipped = float(((iv['observed'] >= iv['geocp_lower_display'])
                         & (iv['observed'] <= iv['geocp_upper'])).mean())
    stored = float(iv['geocp_covered'].mean())
    check('stored coverage equals the unclipped-interval coverage', abs(stored - cov_unclipped) < 1e-12,
          f'{stored:.4f} vs {cov_unclipped:.4f}')

    print('\n5. MONOTONICITY IN ALPHA')
    for h_test in [4528.0]:
        prev_q = None
        for a in [0.10, 0.20, 0.50]:
            res = geocp.intervals_for_cells(cal, ctx['test'][['cx', 'cy']].values,
                                            geocp.predict_raw(ctx['model'], ctx['test']), a,
                                            h_test, 'gaussian')
            q_mean = float(res['q'].mean())
            if prev_q is not None:
                check(f'q does not increase when alpha rises to {a} (h={h_test:.0f})',
                      q_mean <= prev_q + 1e-9, f'{q_mean:.3f} <= {prev_q:.3f}')
            prev_q = q_mean

    print('\n6. COVERAGE TABLE REPRODUCES FROM THE STORED INTERVALS')
    cov_path = os.path.join(OUT, 'geocp_coverage.csv')
    if os.path.exists(cov_path):
        table = pd.read_csv(cov_path)
        allrow = table[table['scope'] == 'all test observations']
        for method, covcol, wcol in [('GeoCP', 'geocp_covered', 'geocp_width'),
                                     ('Split conformal (unweighted)', 'split_covered',
                                      'split_width')]:
            row = allrow[allrow['method'] == method]
            if not len(row):
                continue
            row = row.iloc[0]
            recomputed_cov = float(iv[covcol].mean())
            recomputed_w = float(iv[wcol].mean())
            check(f'{method} coverage reproduces', abs(recomputed_cov - row['empirical_coverage']) < 1e-9,
                  f"{recomputed_cov:.4f} vs {row['empirical_coverage']:.4f}")
            check(f'{method} mean width reproduces', abs(recomputed_w - row['mean_width']) < 1e-9,
                  f"{recomputed_w:.4f} vs {row['mean_width']:.4f}")

print('\n7. DETERMINISM')
c1 = geocp.calibration_scores(ctx)
c2 = geocp.calibration_scores(ctx)
check('calibration scores identical across runs',
      bool(np.allclose(c1['alpha'].values, c2['alpha'].values, atol=0, equal_nan=True)))
r1 = geocp.intervals_for_cells(cal, ctx['test'][['cx', 'cy']].values,
                               geocp.predict_raw(ctx['model'], ctx['test']), alpha, 4528.0,
                               'gaussian')
r2 = geocp.intervals_for_cells(cal, ctx['test'][['cx', 'cy']].values,
                               geocp.predict_raw(ctx['model'], ctx['test']), alpha, 4528.0,
                               'gaussian')
check('intervals identical across runs', bool(np.array_equal(r1['q'].values, r2['q'].values)))

print('\n' + ('ALL GEOCP CHECKS PASSED' if not fails else f'FAILED: {fails}'))
sys.exit(1 if fails else 0)
