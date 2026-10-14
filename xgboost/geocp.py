"""GeoConformal Prediction (GeoCP) for the XGBoost branch.

    python geocp.py                      # default: gaussian kernel, 90% target, h grid search
    python geocp.py --kernel uniform      # sanity limit: must reproduce split conformal
    python geocp.py --alpha 0.10 0.20 0.50

IMPLEMENTED FROM THE THESIS, not from guesses. Every choice below is traceable to the manuscript
(UG_Thesis__Dela_Cruz__Dizon__Dulatre__Perez_.pdf):

* Method: Section 3.5.2. Absolute-residual nonconformity score alpha_i = |y_i - yhat_i| (Eq. 3.36);
  distances in metres from projected coordinates (Eq. 3.40); kernel weights K(d_i / h) (Eq. 3.42)
  normalised to sum to unity (Eq. 3.41/3.43); weighted empirical distribution (Eq. 3.44); local
  weighted quantile (Eq. 3.45); symmetric interval (Eq. 3.46) with L = yhat - q, U = yhat + q
  (Eq. 3.47/3.48) and width W = U - L = 2q (Eq. 3.50).
* Lower bound clipped to zero AFTER the interval is formed, "for visualization"; "The unmodified
  interval should be used when evaluating empirical coverage" (Section 3.5.2) -- so coverage here is
  always computed on the unclipped interval, and clipping only affects the stored display bounds.
* Target coverage: Section 4.5.4 note, "The target coverage level is 1 - alpha (e.g., 90%)".
* Bandwidth: Section 4.5.3, "a grid of candidate values will be evaluated against a spatially
  held-out subset of the calibration partition. The candidate that achieves the target coverage
  probability while minimizing average interval width will be selected as the final bandwidth."
* Calibration must come from cells the model never trained on, and the final spatial test partition
  "remains withheld from all model fitting, hyperparameter tuning, and calibration activities"
  (Sections 4.2.1, 4.5.5).
* Evaluation (Table 4.2): empirical coverage, mean and median interval width, mean
  width-to-prediction ratio, and coverage disaggregated by distance to the nearest calibration
  sensor ("urban core vs peripheral zones") to detect inflation from well-sampled areas.

WHAT THE THESIS DOES NOT SPECIFY, AND WHAT WAS CHOSEN
* The kernel K is named only as "the selected kernel function"/"distance-decaying" (Eq. 3.42,
  Section 3.5.2). No functional form appears anywhere in the manuscript. Default here: **Gaussian**
  K(u) = exp(-u^2 / 2), the standard choice in the geoconformal literature this work follows
  (Lou et al., 2025). `--kernel` also offers `exponential` and `uniform`; `uniform` is not a
  cosmetic option, it is a correctness test (see verify_geocp.py) because in the flat-kernel limit
  the weighted quantile must reduce exactly to split conformal.
* The finite-sample correction is ambiguous in the manuscript: standard split conformal (Eq. 3.36-3.38,
  after Lei et al. 2018) uses the (1-alpha)(1 + 1/m) quantile, while the GeoCP weighted quantile
  (Eq. 3.45) is written at level 1-alpha with no correction. Both are computed and reported
  (`--finite-sample-correction` toggles the primary); with uniform weights they differ by one order
  statistic, which on m = 5,770 records is negligible, but the distinction is stated rather than
  hidden.

HONEST LIMITATIONS, MEASURED BEFORE CODING (probe output in STATUS.md)
* Only 3 calibration cells and 3 test cells exist. Weights are a function of location only.
* The nearest calibration cell is 3,808 m from the nearest test cell (mean 6,283 m), so with three
  calibration sites this far apart the kernel weights are nearly flat for every query cell: this is
  the plan's own warning that "kernel weighting is nearly degenerate and bandwidth selection is
  noisy". geo_effective_sample_size and the weight spread are reported per query cell rather than
  asserted, and the unweighted split-conformal interval is always reported alongside.
* Residuals are strongly cell-clustered (per-cell mean alpha 6.67 / 3.70 / 3.04), which is what the
  weighting is meant to exploit, but with 3 sites the local quantile rests on 3 distinct weight
  levels. This is a pilot demonstration of the machinery, not an interval-quality claim.
"""
import argparse
import json
import os
from math import ceil, sqrt

import numpy as np
import pandas as pd
import xgboost as xgb

import prep_n_proc_pilot as prep
import spatial_cv

OUT_DIR = 'outputs/geocp'
DEFAULT_ALPHAS = (0.10,)
KERNELS = ('gaussian', 'exponential', 'uniform')
# h is in metres. The grid spans "only the single nearest site matters" through "all sites equal",
# which is the full range over which the weighting can behave differently at this network density.
# np.inf is included as a CANDIDATE, matching the CNN-LSTM branch's configs/geocp.yaml
# (bandwidths_m ends in .inf): at infinite bandwidth every calibration record gets equal weight, so
# the weighted quantile reduces exactly to unweighted split conformal. Leaving it in the grid lets the
# selection rule choose "do not weight at all" when the weighted candidates cannot hold coverage --
# which is what this network's three calibration cells need (see STEP4_GEOCP.md).
H_GRID = np.array([500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6500, 8000, 10000,
                   13000, 16000, 20000, 30000, 50000, np.inf], dtype=float)
# How far below the target coverage a candidate may fall and still be selectable, mirroring the
# CNN-LSTM branch's geocp.yaml `coverage_tolerance: 0.02`. 0.0 restores the strict "must reach the
# target on every fold" rule.
DEFAULT_COVERAGE_TOLERANCE = 0.02


# ---------------------------------------------------------------------------------------------
# kernel and weighted quantile
# ---------------------------------------------------------------------------------------------
def kernel_weights(distances, h, kernel='gaussian'):
    """Unnormalised kernel weights K(d / h), Eq. 3.42. d and h are both in metres."""
    u = np.asarray(distances, dtype=float) / float(h)
    if kernel == 'gaussian':
        w = np.exp(-0.5 * u ** 2)
    elif kernel == 'exponential':
        w = np.exp(-u)
    elif kernel == 'uniform':
        w = np.ones_like(u)
    else:
        raise ValueError(f'unknown kernel {kernel!r}; expected one of {KERNELS}')
    return w


def normalised_weights(distances, h, kernel='gaussian', eps=1e-12):
    """Weights normalised to sum to unity, Eq. 3.41/3.43."""
    w = kernel_weights(distances, h, kernel)
    total = w.sum()
    if total <= eps:
        # every weight underflowed: fall back to the single nearest record, which is the limit of a
        # narrow kernel, and report it rather than silently returning an all-zero distribution.
        w = np.zeros_like(w)
        w[int(np.argmin(distances))] = 1.0
        return w
    return w / total


def _effective_level(level, n_records, finite_sample_correction):
    """The weighted-CDF level that weighted_quantile() actually compares against.

    Kept in one place so quantile_grid() cannot drift from weighted_quantile(): the finite-sample
    correction multiplies the level by (1 + 1/m) and the result is capped at 1.0, where m is the number
    of calibration RECORDS contributing weight. Getting this wrong in the grid path silently changed
    every half-width (a 6.57 ug/m3 error at the widest bandwidths).
    """
    if not finite_sample_correction:
        return float(level)
    return min(1.0, float(level) * (1.0 + 1.0 / float(n_records)))


def weighted_quantile(scores, weights, level, finite_sample_correction=True):
    """Smallest score q with weighted CDF >= level, Eq. 3.44/3.45.

    With uniform weights this is the ceil(m * level)-th order statistic, which for
    level = (1-alpha)(1 + 1/m) is the standard split-conformal quantile. Taking the smallest such q
    (rather than interpolating) is what makes the empirical coverage at least the requested level.
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = scores.size
    if m == 0:
        raise ValueError('empty calibration set')
    level = _effective_level(level, m, finite_sample_correction)
    order = np.argsort(scores, kind='mergesort')
    s_sorted = scores[order]
    w_sorted = weights[order]
    cdf = np.cumsum(w_sorted)
    cdf[-1] = 1.0                       # guard against float drift
    idx = int(np.searchsorted(cdf, level - 1e-12, side='left'))
    idx = min(idx, m - 1)
    return float(s_sorted[idx])


# ---------------------------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------------------------
def load_context(fold_json='fold_partitions.json'):
    """Training/calibration/test rows, the fitted production model, and cell centroids."""
    doc = spatial_cv.load_fold_map(fold_json)
    parts = doc['partitions']
    centroids = spatial_cv.get_grid_centroids().set_index('Grid_ID')

    df = prep.df_clean.copy()
    df['cx'] = df['Grid_ID'].map(centroids['cx'])
    df['cy'] = df['Grid_ID'].map(centroids['cy'])
    train = df[df['Grid_ID'].isin(parts['train'])].reset_index(drop=True)
    cal = df[df['Grid_ID'].isin(parts['calibration'])].reset_index(drop=True)
    test = df[df['Grid_ID'].isin(parts['test'])].reset_index(drop=True)

    results = pd.read_csv(os.path.join('outputs', 'ablation_results.csv'))
    row = results[(results['arm'] == 'C')].set_index('outer_fold')
    best_fold = int(row['r2'].idxmax())
    params = json.loads(row.loc[best_fold, 'best_params'])
    n_est = int(row.loc[best_fold, 'n_estimators'])

    model = xgb.XGBRegressor(n_estimators=n_est, random_state=42, tree_method='hist', n_jobs=-1,
                             **params)
    model.fit(train[prep.features].values, np.log1p(train['Value'].values))
    return {'parts': parts, 'train': train, 'cal': cal, 'test': test, 'model': model,
            'params': params, 'n_estimators': n_est, 'best_fold': best_fold,
            'centroids': centroids, 'fold_doc': doc}


def predict_raw(model, frame):
    """Point predictions back in raw ug/m3 (the model is fitted on log1p)."""
    return np.expm1(model.predict(frame[prep.features].values))


def calibration_scores_for_model(model, df, grid_ids, mapping_csv_path='sensor_to_grid_map_full.csv',
                                 centroids=None):
    """Georeferenced calibration table for an ALREADY-FITTED model and an explicit cell list.

    This is the entry point for the notebook, which fits the production model in its own cells and
    then calls GeoCP as a post-prediction layer. It mirrors calibration_scores() exactly; the only
    difference is that the model and the panels are passed in rather than rebuilt, so the notebook
    does not refit the model a second time (which is what load_context() would do).
    """
    if centroids is None:
        centroids = spatial_cv.get_grid_centroids(mapping_csv_path).set_index('Grid_ID')
    panel = df[df['Grid_ID'].isin(list(grid_ids))].reset_index(drop=True).copy()
    panel['cx'] = panel['Grid_ID'].map(centroids['cx'])
    panel['cy'] = panel['Grid_ID'].map(centroids['cy'])
    pred = predict_raw(model, panel)
    return pd.DataFrame({
        'Location': panel['Location'].values,
        'Grid_ID': panel['Grid_ID'].values,
        'DateTime': panel['DateTime'].values,
        'observed': panel['Value'].values,
        'predicted': pred,
        'alpha': np.abs(panel['Value'].values - pred),
        'cx': panel['cx'].values,
        'cy': panel['cy'].values,
    })


def calibration_scores(ctx):
    """Georeferenced calibration table: alpha_i, coordinates, cell id.

    Thin wrapper around calibration_scores_for_model() for the CLI path, where load_context() has
    already fitted the model and assembled the panels.
    """
    return calibration_scores_for_model(ctx['model'], ctx['cal'], ctx['cal']['Grid_ID'].unique(),
                                       centroids=ctx['centroids'])


def distance_matrix(query_xy, cal_xy):
    """Euclidean distances in metres, Eq. 3.40."""
    q = np.asarray(query_xy, dtype=float)[:, None, :]
    c = np.asarray(cal_xy, dtype=float)[None, :, :]
    return np.sqrt(((q - c) ** 2).sum(axis=-1))


def intervals_for_cells(cal_scores, query_xy, model_pred, alpha, h, kernel,
                        finite_sample_correction=True):
    """GeoCP intervals for every query row, using its cell's coordinates.

    Returns a frame with prediction, q, lower/upper (unclipped), the display lower bound, width, and
    two degeneracy diagnostics, both computed per CALIBRATION CELL rather than per calibration
    record: `top_cell_weight` is the share of total kernel weight landing on the single most heavily
    weighted calibration cell (near 1.0 means the interval rests on one site, near 1/C means the
    sites are effectively pooled) and `effective_n_cells` is the inverse Simpson index in units of
    cells. Record-level weights are misleading here because each cell contributes thousands of
    records, so no single record can ever carry much weight.
    """
    d = distance_matrix(query_xy, cal_scores[['cx', 'cy']].values)
    scores = cal_scores['alpha'].values
    cell_ids = cal_scores['Grid_ID'].values
    q_vals = np.empty(len(query_xy))
    top_cell = np.empty(len(query_xy))
    eff_cells = np.empty(len(query_xy))
    for i in range(len(query_xy)):
        w = normalised_weights(d[i], h, kernel)
        q_vals[i] = weighted_quantile(scores, w, 1.0 - alpha, finite_sample_correction)
        per_cell = pd.Series(w).groupby(cell_ids).sum().values
        top_cell[i] = float(np.max(per_cell))
        eff_cells[i] = 1.0 / float(np.sum(per_cell ** 2))
    return pd.DataFrame({
        'prediction': model_pred,
        'q': q_vals,
        'lower': model_pred - q_vals,
        'upper': model_pred + q_vals,
        'lower_display': np.maximum(model_pred - q_vals, 0.0),
        'width': 2.0 * q_vals,
        'top_cell_weight': top_cell,
        'effective_n_cells': eff_cells,
    })


def coverage(actual, lower, upper):
    """Empirical coverage on the UNCLIPPED interval."""
    return float(np.mean((actual >= lower) & (actual <= upper)))


def quantile_grid(cal_scores, query_xy, h, kernel='gaussian', level=0.90, batch_size=20000,
                  finite_sample_correction=True, width_cap=None, effective_level=None):
    """Exact weighted quantile at many query points at once.

    Same definition as weighted_quantile (Eq. 3.44/3.45: the smallest calibration score whose weighted
    CDF reaches `level`), evaluated for every query point. Ported in spirit from the CNN-LSTM branch's
    GeoCP.quantiles_grid, which exists so that intervals can be produced for EVERY raster cell rather
    than per station.

    Implemented as the reference computation itself -- per record, kernel weight, sort by score, find
    where the normalised cumulative weight first reaches the effective level -- rather than as a
    per-cell merge walk. An earlier per-cell walk version produced half-widths that disagreed with
    intervals_for_cells on 3,568 of 5,420 test rows (up to 1.68 ug/m3), because the finite-sample
    correction divides by the number of contributing RECORDS, which a per-cell merge does not track.
    Correctness beats cleverness here: this runs in about a second on the pilot (3 cells, 5,770
    calibration records). If a 10 m city-wide raster ever makes the O(m) sweep the bottleneck, the
    per-cell merge can be reintroduced behind a test that compares it against this one on a subsample.
    """
    scores = np.asarray(cal_scores['alpha'].values, dtype=float)
    cal_xy = np.asarray(cal_scores[['cx', 'cy']].values, dtype=float)
    query_xy = np.asarray(query_xy, dtype=float).reshape(-1, 2)
    m = len(scores)
    order = np.argsort(scores, kind='mergesort')
    s_sorted = scores[order]
    threshold = (float(effective_level) if effective_level is not None
                 else _effective_level(level, m, finite_sample_correction))

    out = np.empty(len(query_xy), dtype=float)
    for start in range(0, len(query_xy), batch_size):
        part = query_xy[start:start + batch_size]
        dist = np.sqrt(((part[:, None, :] - cal_xy[None, :, :]) ** 2).sum(axis=-1))   # [b, m]
        for i in range(len(part)):
            # normalised_weights() is reused rather than re-deriving weights here. It carries the
            # underflow guard that matters at narrow bandwidths: when every K(d/h) underflows to 0 it
            # falls back to the single nearest record. Re-implementing that inline is exactly how this
            # function first diverged from intervals_for_cells (3,568 of 5,420 rows wrong).
            w = normalised_weights(dist[i], h, kernel)
            cdf = np.cumsum(w[order])
            hit = int(np.searchsorted(cdf, threshold - 1e-12, side='left'))
            out[start + i] = np.inf if hit >= m else float(s_sorted[hit])
    return out


def resolve_bandwidth(cal_scores, alpha, kernel, h_grid, h_mode, h_fixed, finite_sample_correction,
                      query_xy=None):
    """Return (h, curve, rule, degraded) for the requested bandwidth mode.

    'thesis-grid'    Section 4.5.3 grid search on the spatially held-out calibration subset.
    'median-distance' h = median distance from the query partition's cells to their nearest
                     calibration observation. Geometry only: no labels are used, and it cannot
                     over-concentrate the weighting on a single calibration site the way a very
                     narrow grid winner can. Reported as a sensitivity check on the grid search.
    'fixed'          use --h verbatim.
    """
    if h_mode == 'fixed':
        if h_fixed is None:
            raise ValueError('--h-mode fixed requires --h')
        return float(h_fixed), pd.DataFrame(), 'fixed by --h (bandwidth selection skipped)', False

    if h_mode == 'median-distance':
        if query_xy is None:
            raise ValueError('median-distance mode needs the query coordinates')
        d = distance_matrix(np.asarray(query_xy, dtype=float),
                            cal_scores[['cx', 'cy']].values).min(axis=1)
        chosen = float(np.median(d))
        rule = (f'h = median distance from the query cells to their nearest calibration '
                f'observation ({chosen:.0f} m); geometry only, no labels used')
        return chosen, pd.DataFrame(), rule, False

    chosen_h, curve, rule = select_bandwidth(cal_scores, alpha, kernel, h_grid,
                                             finite_sample_correction)
    target = 1.0 - alpha
    reached = bool((curve['worst_fold_coverage'] >= target - DEFAULT_COVERAGE_TOLERANCE).any())
    degraded = not reached
    if degraded:
        rule = (rule + ' -- DEGRADED: no bandwidth reached the target coverage on every '
                       'calibration fold, so the selected bandwidth is not coverage-valid. Report '
                       'the unweighted split-conformal interval as the primary calibrated result '
                       'and GeoCP as a pilot.')
    elif np.isinf(chosen_h):
        degraded = False
        rule = (rule + ' -- note: the selected bandwidth is infinite, i.e. the spatially weighted '
                       'quantile fell back to unweighted split conformal. Report that as the finding '
                       'rather than presenting a weighted interval.')
    return chosen_h, curve, rule, degraded


def select_bandwidth(cal_scores, alpha, kernel, h_grid, finite_sample_correction=True,
                     target_tolerance=DEFAULT_COVERAGE_TOLERANCE):
    """Section 4.5.3: grid search on a spatially held-out subset of the calibration partition.

    Leave-one-calibration-cell-out: build the weights from the other two cells, then measure coverage
    and width on the held-out cell. Selection rule, stated explicitly because "achieves the target
    coverage while minimising width" is ambiguous when folds disagree:
      among h whose WORST-fold coverage is within `target_tolerance` of 1-alpha, pick the smallest
      mean width; if no h reaches the target on every fold, pick the largest worst-fold coverage.
    A permissive rule would be to take the mean coverage across folds; the worst-fold rule is the
    conservative reading and is the one reported.
    """
    cells = sorted(cal_scores['Grid_ID'].unique())
    rows = []
    # Scores can be infinite when a fold has no calibration weight near the held-out cell; those rows
    # count as covered but are capped when averaging widths, so they neither dominate the width
    # comparison nor get treated as free. Same convention as the CNN-LSTM branch's select_config().
    width_cap = 2.0 * float(cal_scores['alpha'].max())
    for h in h_grid:
        fold_cov, fold_w, fold_wmed = [], [], []
        for held in cells:
            fit = cal_scores[cal_scores['Grid_ID'] != held]
            sel = cal_scores[cal_scores['Grid_ID'] == held]
            if len(fit) == 0 or len(sel) == 0:
                continue
            res = intervals_for_cells(fit, sel[['cx', 'cy']].values, sel['predicted'].values,
                                      alpha, h, kernel, finite_sample_correction)
            fold_cov.append(coverage(sel['observed'].values, res['lower'].values,
                                     res['upper'].values))
            q = res['q'].values
            fold_w.append(float(np.mean(np.minimum(q, width_cap)) * 2.0))
            fold_wmed.append(float(np.median(np.minimum(q, width_cap)) * 2.0))
        if not fold_cov:
            continue
        rows.append({'h': float(h), 'kernel': kernel, 'alpha': alpha,
                     'worst_fold_coverage': min(fold_cov), 'mean_fold_coverage': float(np.mean(fold_cov)),
                     'mean_width': float(np.mean(fold_w)), 'median_width': float(np.mean(fold_wmed)),
                     'fold_coverages': ';'.join(f'{c:.4f}' for c in fold_cov)})

    curve = pd.DataFrame(rows)
    target = 1.0 - alpha
    # Candidates that reach the target within `target_tolerance`, i.e. the coverage tolerance from the
    # CNN-LSTM branch's geocp.yaml. Among those, the narrowest wins; np.inf is a candidate like any
    # other, so "unweighted split conformal" can be selected outright.
    ok = curve[curve['worst_fold_coverage'] >= target - target_tolerance]
    if len(ok):
        chosen = ok.sort_values(['mean_width', 'h']).iloc[0]
        if np.isinf(chosen['h']):
            rule = (f'unweighted split conformal selected: at infinite bandwidth every calibration '
                    f'record is equally weighted, and it is the narrowest candidate within '
                    f'{target_tolerance:.2f} of the target coverage on every fold')
        else:
            rule = (f'smallest mean width among h whose worst-fold coverage is within '
                    f'{target_tolerance:.2f} of the target')
    else:
        chosen = curve.sort_values(['worst_fold_coverage', 'mean_width'],
                                   ascending=[False, True]).iloc[0]
        rule = (f'no h reached the target coverage within {target_tolerance:.2f} on every fold; '
                f'chose the largest worst-fold coverage (report as an under-covering pilot)')
    curve['selected'] = curve['h'] == chosen['h']
    return float(chosen['h']), curve, rule


def evaluate(ctx, cal_scores, alpha, h, kernel, finite_sample_correction=True, model=None):
    """GeoCP plus the unweighted split-conformal baseline on the final spatial test partition.

    `model` defaults to the one inside ctx (the CLI path, where load_context() fitted it). The
    notebook passes its own already-fitted production_model instead, so the model is not refitted.
    """
    test = ctx['test']
    if model is None:
        model = ctx['model']
    pred = predict_raw(model, test)
    actual = test['Value'].values
    xy = test[['cx', 'cy']].values

    geo = intervals_for_cells(cal_scores, xy, pred, alpha, h, kernel, finite_sample_correction)

    # unweighted split conformal: the flat-kernel limit, reported alongside per the plan
    uniform_q = weighted_quantile(cal_scores['alpha'].values,
                                  np.full(len(cal_scores), 1.0 / len(cal_scores)),
                                  1.0 - alpha, finite_sample_correction)
    split = pd.DataFrame({
        'prediction': pred, 'q': uniform_q, 'lower': pred - uniform_q, 'upper': pred + uniform_q,
        'lower_display': np.maximum(pred - uniform_q, 0.0), 'width': 2.0 * uniform_q,
        'top_cell_weight': np.nan, 'effective_n_cells': np.nan,
    })

    # distance to the nearest calibration observation, for the urban-core / peripheral split
    d_cal = distance_matrix(xy, cal_scores[['cx', 'cy']].values).min(axis=1)
    out = pd.DataFrame({
        'Location': test['Location'].values, 'Grid_ID': test['Grid_ID'].values,
        'DateTime': test['DateTime'].values, 'observed': actual, 'prediction': pred,
        'dist_to_nearest_cal_m': d_cal,
        'geocp_q': geo['q'].values, 'geocp_lower': geo['lower'].values,
        'geocp_upper': geo['upper'].values, 'geocp_lower_display': geo['lower_display'].values,
        'geocp_width': geo['width'].values,
        'geocp_top_cell_weight': geo['top_cell_weight'].values,
        'geocp_effective_n_cells': geo['effective_n_cells'].values,
        'split_q': split['q'].values, 'split_lower': split['lower'].values,
        'split_upper': split['upper'].values, 'split_width': split['width'].values,
    })
    out['geocp_covered'] = (out['observed'] >= out['geocp_lower']) & \
                           (out['observed'] <= out['geocp_upper'])
    out['split_covered'] = (out['observed'] >= out['split_lower']) & \
                           (out['observed'] <= out['split_upper'])
    out['partition_side'] = np.where(out['dist_to_nearest_cal_m'] <= np.median(d_cal),
                                     'nearer half (urban core)', 'farther half (peripheral)')
    return out


def summarise(intervals, alpha, h, kernel, rule):
    """Table 4.2 metrics, globally and split by distance to the nearest calibration sensor."""
    rows = []
    med_d = float(np.median(intervals['dist_to_nearest_cal_m']))

    def block(name, sub):
        for method, cov, lo, hi, w in [
            ('GeoCP', 'geocp_covered', 'geocp_lower', 'geocp_upper', 'geocp_width'),
            ('Split conformal (unweighted)', 'split_covered', 'split_lower', 'split_upper',
             'split_width'),
        ]:
            rows.append({
                'scope': name, 'method': method, 'n': len(sub),
                'empirical_coverage': float(sub[cov].mean()),
                'mean_width': float(sub[w].mean()),
                'median_width': float(sub[w].median()),
                'mean_width_to_prediction': float((sub[w] / sub['prediction']).mean()),
                'mean_abs_error': float(np.abs(sub['observed'] - sub['prediction']).mean()),
            })

    block('all test observations', intervals)
    block('nearer half (urban core)', intervals[intervals['dist_to_nearest_cal_m'] <= med_d])
    block('farther half (peripheral)', intervals[intervals['dist_to_nearest_cal_m'] > med_d])
    for cell, sub in intervals.groupby('Grid_ID'):
        block(f'cell {int(cell)}', sub)

    summary = pd.DataFrame(rows)
    summary['alpha'] = alpha
    summary['nominal_coverage'] = 1.0 - alpha
    summary['h_m'] = h
    summary['kernel'] = kernel
    summary['bandwidth_rule'] = rule
    return summary


def run(alphas=DEFAULT_ALPHAS, kernel='gaussian', h=None, finite_sample_correction=True,
        out_dir=OUT_DIR, h_mode='thesis-grid'):
    ctx = load_context()
    os.makedirs(out_dir, exist_ok=True)
    cal_scores = calibration_scores(ctx)
    cal_scores.to_csv(os.path.join(out_dir, 'geocp_calibration_scores.csv'), index=False)

    print('=== calibration nonconformity scores (alpha_i = |y - yhat|, ug/m3) ===')
    print(cal_scores.groupby('Grid_ID')['alpha']
          .agg(['size', 'mean', 'median', 'std', lambda s: s.quantile(0.9)])
          .rename(columns={'<lambda_0>': 'p90'}).round(3).to_string())

    results = []
    curves = []
    for alpha in alphas:
        print(f'\n=== alpha = {alpha} (target coverage {1 - alpha:.0%}), kernel = {kernel}, '
              f'h-mode = {h_mode} ===')
        chosen_h, curve, rule, degraded = resolve_bandwidth(
            cal_scores, alpha, kernel, H_GRID, h_mode, h,
            finite_sample_correction, query_xy=ctx['test'][['cx', 'cy']].values)
        if len(curve):
            curve['alpha'] = alpha
            curves.append(curve)
            print(f'  bandwidth selected: h = {chosen_h:.0f} m')
            print(f'  rule: {rule}')
            print(curve[['h', 'worst_fold_coverage', 'mean_fold_coverage', 'mean_width']]
                  .round(3).to_string(index=False))
        else:
            print(f'  bandwidth: h = {chosen_h:.0f} m')
            print(f'  rule: {rule}')
        if degraded:
            print('  !! DEGRADED: the selected bandwidth did not reach the target coverage on the')
            print('     held-out calibration cells, so GeoCP is not coverage-valid here. The')
            print('     unweighted split-conformal interval is the primary result for this alpha.')

        intervals = evaluate(ctx, cal_scores, alpha, chosen_h, kernel, finite_sample_correction)
        intervals['alpha'] = alpha
        intervals.to_csv(os.path.join(out_dir, f'geocp_test_intervals_alpha{int(alpha*100)}.csv'),
                         index=False)
        summary = summarise(intervals, alpha, chosen_h, kernel, rule)
        summary['degraded_bandwidth'] = degraded
        results.append(summary)

        print('\n  --- evaluation on the final spatial test partition (unclipped intervals) ---')
        top = summary[summary['scope'] == 'all test observations']
        print(top[['method', 'empirical_coverage', 'mean_width', 'median_width',
                   'mean_width_to_prediction']].round(3).to_string(index=False))
        print('\n  --- coverage by distance to the nearest calibration sensor ---')
        print(summary[summary['scope'] != 'all test observations']
              [['scope', 'method', 'n', 'empirical_coverage', 'mean_width']]
              .round(3).to_string(index=False))
        n_cells = cal_scores['Grid_ID'].nunique()
        top_w = intervals['geocp_top_cell_weight']
        eff_c = intervals['geocp_effective_n_cells']
        print(f'\n  calibration-cell weight concentration: the most heavily weighted single cell '
              f'carries {top_w.mean():.3f} of the weight on average (max {top_w.max():.3f}), and '
              f'the effective number of calibration cells is {eff_c.mean():.2f} of {n_cells}')
        print('  -> an effective count near 1 means the interval rests on a single calibration '
              'site;')
        print(f'     near {n_cells} means the sites are effectively pooled. State which regime the '
              f'selected')
        print('     bandwidth is in, because "geographically weighted" is not informative when it '
              'is really one site.')

    summary_all = pd.concat(results, ignore_index=True)
    summary_all.to_csv(os.path.join(out_dir, 'geocp_coverage.csv'), index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(
            os.path.join(out_dir, 'geocp_bandwidth_curve.csv'), index=False)

    cfg = {
        'implemented_from': 'thesis Section 3.5.2 (method) and 4.5.3-4.5.5 (implementation)',
        'kernel': kernel,
        'alphas': list(alphas),
        'nominal_coverage': [1 - a for a in alphas],
        'bandwidth_m': {str(a): (None if h is None else float(h)) for a in alphas},
        'kernel_choice_note': ('the manuscript does not name a kernel function; Gaussian '
                               'exp(-u^2/2) chosen as the standard geoconformal choice'),
        'finite_sample_correction': finite_sample_correction,
        'lower_bound_policy': ('clipped to 0 for display only; coverage always on the unclipped '
                               'interval, per Section 3.5.2'),
        'calibration_cells': ctx['parts']['calibration'],
        'test_cells': ctx['parts']['test'],
        'train_cells': ctx['parts']['train'],
        'model': {'params': ctx['params'], 'n_estimators': ctx['n_estimators'],
                  'params_from_outer_fold': ctx['best_fold']},
        'n_calibration_records': int(len(cal_scores)),
        'n_test_records': int(len(ctx['test'])),
    }
    with open(os.path.join(out_dir, 'geocp_config.json'), 'w', encoding='utf-8') as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
    print(f'\nwrote {out_dir}/: calibration scores, intervals per alpha, coverage table, '
          f'bandwidth curve, config')
    return summary_all


def main():
    ap = argparse.ArgumentParser(description='GeoConformal Prediction for the XGBoost branch')
    ap.add_argument('--alpha', nargs='+', type=float, default=list(DEFAULT_ALPHAS))
    ap.add_argument('--kernel', default='gaussian', choices=KERNELS)
    ap.add_argument('--h-mode', default='thesis-grid',
                    choices=('thesis-grid', 'median-distance', 'fixed'),
                    help='thesis-grid = Section 4.5.3 grid search; median-distance = geometry-only '
                         'alternative; fixed = use --h')
    ap.add_argument('--h', type=float, default=None,
                    help='bandwidth in metres, used when --h-mode fixed')
    ap.add_argument('--no-finite-sample-correction', action='store_true')
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    run(alphas=tuple(args.alpha), kernel=args.kernel, h=args.h,
        finite_sample_correction=not args.no_finite_sample_correction, out_dir=args.out_dir,
        h_mode=args.h_mode)


if __name__ == '__main__':
    main()
