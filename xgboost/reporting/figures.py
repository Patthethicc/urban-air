"""Phase 5: the report figures, written from the CLI (not the notebook).

The CNN-LSTM branch writes its figures to `plots/` from `reporting/plotters.py` during a normal run,
which is why they exist without anyone opening a notebook. This does the same, so `--stage figures`
(or `--stage all`) leaves a figure set behind:

    pred_vs_obs.png          predicted against observed, test stations
    residual_hist.png        distribution of actual - predicted
    accuracy_by_station.png  RMSE of every test station
    timeseries_sample.png    a few days of observed vs predicted for one station
    error_vs_concentration.png   how error grows with concentration (the tail problem)
    geocp_coverage.png       interval coverage per test station (only if GeoCP has run)

Every figure is skipped with a printed reason when its input is missing, so this never becomes the
reason a pipeline run fails.

    python -m reporting.figures
"""
import json
import os

import numpy as np
import pandas as pd

import data

OUT = os.path.join(data.OUT_DIR, 'figures')
PRED = os.path.join(data.OUT_DIR, 'predictions_test.csv')


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(fig, name, written):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    import matplotlib.pyplot as plt
    plt.close(fig)
    written.append(path)
    print('  %-28s %s' % (name, 'ok'))


def pred_vs_obs(d, written):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(d.actual, d.predicted, s=3, alpha=0.15, edgecolors='none')
    lim = [0, float(np.nanpercentile(d.actual, 99.5))]
    ax.plot(lim, lim, color='#c0392b', lw=1, label='1:1')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('observed PM2.5 (µg/m³)'); ax.set_ylabel('predicted (µg/m³)')
    r2 = 1 - np.sum((d.actual - d.predicted) ** 2) / np.sum((d.actual - d.actual.mean()) ** 2)
    ax.set_title('Test stations: predicted vs observed  (R² %.3f)' % r2, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8)
    _save(fig, 'pred_vs_obs.png', written)


def residual_hist(d, written):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(d.residual.clip(-25, 25), bins=80, color='#4a7fb5')
    ax.axvline(0, color='#c0392b', lw=1)
    ax.set_xlabel('residual = actual − predicted (µg/m³)'); ax.set_ylabel('rows')
    ax.set_title('Residuals, test stations (clipped at ±25 for display)', fontsize=10, loc='left')
    _save(fig, 'residual_hist.png', written)


def accuracy_by_station(d, written):
    plt = _mpl()
    g = d.groupby('Location').apply(
        lambda x: pd.Series({'rmse': float(np.sqrt(np.mean(x.residual ** 2))),
                             'n': int(len(x))}), include_groups=False)
    g = g.sort_values('rmse')
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(range(len(g)), g['rmse'], color='#4a7fb5')
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(i) for i in g.index], rotation=90, fontsize=7)
    ax.set_ylabel('RMSE (µg/m³)'); ax.set_xlabel('test station id')
    ax.set_title('Accuracy by test station (lowest to highest)', fontsize=10, loc='left')
    _save(fig, 'accuracy_by_station.png', written)
    return g


def timeseries_sample(d, written):
    plt = _mpl()
    loc = d.groupby('Location').size().idxmax()
    s = d[d.Location == loc].sort_values('DateTime')
    s = s[(s.DateTime >= s.DateTime.min()) & (s.DateTime <= s.DateTime.min() + pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.plot(s.DateTime, s.actual, lw=1.0, label='observed', color='#2c3e50')
    ax.plot(s.DateTime, s.predicted, lw=1.0, label='predicted', color='#c0392b', alpha=0.85)
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_title('Station %s, first week of the test period' % loc, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8)
    _save(fig, 'timeseries_sample.png', written)


def error_vs_concentration(d, written):
    plt = _mpl()
    bins = [0, 10, 20, 30, 40, 60, 80, 120, 1e9]
    labels = ['0-10', '10-20', '20-30', '30-40', '40-60', '60-80', '80-120', '>120']
    d = d.assign(band=pd.cut(d.actual, bins=bins, labels=labels, right=False))
    g = d.groupby('band', observed=True).apply(
        lambda x: pd.Series({'rmse': float(np.sqrt(np.mean(x.residual ** 2))),
                             'bias': float(x.residual.mean()),
                             'n': int(len(x))}), include_groups=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(g))
    ax.bar(x, g['rmse'], color='#4a7fb5', label='RMSE')
    ax.plot(x, g['bias'], 'o-', color='#c0392b', lw=1, ms=4, label='bias')
    ax.set_xticks(x); ax.set_xticklabels(list(g.index), fontsize=8)
    ax.set_xlabel('observed concentration band (µg/m³)'); ax.set_ylabel('µg/m³')
    ax.set_title('Error and bias by concentration — where the RMSE comes from',
                 fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8)
    _save(fig, 'error_vs_concentration.png', written)
    return g


def geocp_coverage(written):
    plt = _mpl()
    # Prefer the shared-profile run: it is the one comparable to the CNN-LSTM and the one quoted in
    # the results table. `outputs/geocp/` is the branch's own single-kernel grid and uses a different
    # calibration subsample, so citing it here made the figure and the table describe different runs.
    candidates = [os.path.join(data.OUT_DIR, 'geocp_shared', 'predictions_test_intervals.csv'),
                  os.path.join(data.OUT_DIR, 'geocp', 'geocp_test_intervals_alpha10.csv')]
    cov_file = next((p for p in candidates if os.path.exists(p)), None)
    if cov_file is None:
        print('  geocp_coverage.png           skipped (run --stage geocp first)')
        return
    print('  geocp_coverage.png           source: %s' % os.path.basename(os.path.dirname(cov_file)))
    iv = pd.read_csv(cov_file)
    # geocp.py writes `geocp_covered` / `split_covered`; older frames may only carry the bounds
    if 'geocp_covered' in iv.columns:
        iv['covered'] = iv['geocp_covered']
    elif 'covered' in iv.columns:
        pass
    elif {'observed', 'geocp_lower', 'geocp_upper'} <= set(iv.columns):
        iv['covered'] = (iv['observed'] >= iv['geocp_lower']) & (iv['observed'] <= iv['geocp_upper'])
    else:
        print('  geocp_coverage.png           skipped (interval file has no coverage column)')
        return
    g = iv.groupby('Location')['covered'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(range(len(g)), g.values * 100, color='#4a7fb5')
    ax.axhline(90, color='#c0392b', lw=1, ls='--', label='90% target')
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(i) for i in g.index], rotation=90, fontsize=7)
    ax.set_ylabel('coverage (%)'); ax.set_xlabel('test station id')
    ax.set_title('Interval coverage by test station', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8)
    _save(fig, 'geocp_coverage.png', written)


FAMILIES = {
    'morphology': ['bh_mean', 'bh_stdev', 'ndvi_mean', 'bldg_dens', 'aspect_ratio_mean',
                   'street_width_mean'],
    'time': ['Hour', 'hour_sin', 'hour_cos', 'DayOfWeek', 'Month', 'dayofweek_sin', 'dayofweek_cos',
             'month_sin', 'month_cos'],
    'pm history': ['pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_4', 'pm25_roll24_mean',
                   'pm25_roll24_std', 'pm25_roll72_mean', 'pm25_roll72_std', 'pm25_diff_1_2'],
    'interaction': ['hour_x_bldg_dens', 'hour_x_ndvi', 'hour_x_aspect_ratio'],
    'spatial lag': ['pm25_spatial_lag', 'pm25_spatial_lag_w', 'pm25_spatial_lag_n'],
    'patch texture': [],      # filled by pattern below
}


def _family_of(name):
    for fam, cols in FAMILIES.items():
        if name in cols:
            return fam
    if name.startswith('tex_'):
        return 'patch texture'
    return 'other'


def shap_importance(written, arm="C", n_sample=4000, seed=42):
    """Mean |SHAP| per feature, grouped by family, for the FULL model (arm C by default).

    The attribution claim ("morphology carries ~2.5%") is about the full model, but the model that gets
    DEPLOYED is whichever arm won the ablation -- and in the current run that is arm A, which contains no
    morphology feature at all, so a SHAP run on the artifact reports 0.0% and proves nothing. This fits
    the requested arm from the ablation hyperparameters instead and labels the figure with it.
    """
    plt = _mpl()
    import joblib
    import shap as _shap

    import xgboost as xgb
    from preprocessing import prep
    from preprocessing import spatial_cv
    from training.ablation import ARM_FEATURES, build_design_matrix
    from training.final_test import tuned_config

    doc = spatial_cv.load_fold_map()
    df = prep.df_clean
    tr = df[df['Grid_ID'].isin(doc['partitions']['train'])].reset_index(drop=True)
    if len(tr) > n_sample:
        tr = tr.sample(n_sample, random_state=seed).reset_index(drop=True)

    params, n_est, src = tuned_config(arm)
    feats = list(ARM_FEATURES[arm])
    model = xgb.XGBRegressor(n_estimators=n_est, random_state=42, tree_method='hist',
                             n_jobs=-1, **params)
    model.fit(build_design_matrix(tr, arm), np.log1p(tr["Value"].to_numpy(float)))
    X = build_design_matrix(tr, arm)

    expl = _shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1]
    names = [feats[i] for i in order]
    vals = imp[order]

    fam_tot = {}
    for n_, v_ in zip(names, vals):
        fam_tot[_family_of(n_)] = fam_tot.get(_family_of(n_), 0.0) + float(v_)
    total = sum(fam_tot.values()) or 1.0
    fam_pct = {k: 100.0 * v / total for k, v in sorted(fam_tot.items(), key=lambda kv: -kv[1])}

    colours = {'pm history': '#2c3e50', 'time': '#4a7fb5', 'morphology': '#c0392b',
               'interaction': '#e67e22', 'patch texture': '#27ae60', 'spatial lag': '#8e44ad',
               'other': '#95a5a6'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={'width_ratios': [1.55, 1.0]})
    ax1.barh(range(len(names))[::-1], vals, color=[colours.get(_family_of(n_), '#95a5a6') for n in names])
    ax1.set_yticks(range(len(names))[::-1])
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlabel('mean |SHAP| (µg/m³)')
    ax1.set_title('Feature attribution, arm %s (%d rows)' % (arm, len(X)), fontsize=10, loc='left')

    fams = list(fam_pct)
    ax2.barh(range(len(fams))[::-1], [fam_pct[f] for f in fams],
             color=[colours.get(f, '#95a5a6') for f in fams])
    ax2.set_yticks(range(len(fams))[::-1])
    ax2.set_yticklabels(fams, fontsize=8)
    for i, f in enumerate(fams):
        ax2.text(fam_pct[f] + 1.0, len(fams) - 1 - i, '%.1f%%' % fam_pct[f], va='center', fontsize=7)
    ax2.set_xlabel('% of total attribution')
    ax2.set_title('By feature family', fontsize=10, loc='left')
    for a in (ax1, ax2):
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
    _save(fig, 'shap_importance.png', written)

    # ---- beeswarm: direction and spread, not just magnitude. This is the figure a CNN would answer
    # with Grad-CAM; for a tree ensemble TreeSHAP is the exact equivalent (no gradients involved).
    try:
        import shap as _shap2
        plt2 = _mpl()
        fig2 = plt2.figure(figsize=(7.4, 5.0))
        _shap2.summary_plot(sv, X, feature_names=feats, show=False, max_display=15)
        fig2 = plt2.gcf()
        fig2.set_size_inches(7.4, 5.0)
        plt2.title('SHAP value (log1p scale) — impact on the prediction, arm %s' % arm,
                   fontsize=10, loc='left')
        _save(fig2, 'shap_beeswarm.png', written)
    except Exception as exc:
        print('  shap_beeswarm.png          skipped (%s)' % exc)

    # ---- waterfall: ONE prediction, explained feature by feature (the local counterpart)
    try:
        import shap as _shap3
        import xgboost as _xgb
        row_pred = model.predict(X)
        k = int(np.argmax(row_pred))          # the worst-case (highest predicted) row in the sample
        expl = _shap3.Explanation(
            values=sv[k], base_values=expl.expected_value, data=X[k],
            feature_names=feats)
        plt3 = _mpl()
        fig3 = plt3.figure(figsize=(7.6, 5.2))
        _shap3.plots.waterfall(expl, max_display=12, show=False)
        fig3 = plt3.gcf()
        fig3.set_size_inches(7.6, 5.2)
        _save(fig3, 'shap_waterfall.png', written)
        json.dump({'row_index': k, 'predicted_log1p': float(row_pred[k]),
                   'expm1_ugm3': float(np.expm1(row_pred[k])), 'arm': arm},
                  open(os.path.join(OUT, 'shap_waterfall.json'), 'w'), indent=2)
    except Exception as exc:
        print('  shap_waterfall.png         skipped (%s)' % exc)

    json.dump({'arm': arm, 'families_pct': fam_pct, 'n_rows': int(len(X)),
               'top_features': names[:12], 'params_source': src},
              open(os.path.join(OUT, 'shap_importance.json'), 'w'), indent=2)
    print('       morphology share: %.1f%%' % fam_pct.get('morphology', 0.0))
    return fam_pct

def main(arm=None):
    global OUT
    base = OUT
    if arm:
        OUT = os.path.join(base, str(arm))     # keep every arm's set instead of overwriting
    os.makedirs(OUT, exist_ok=True)
    print('=== figures -> %s ===' % OUT)
    src = PRED
    if arm:
        per_arm = os.path.join(data.OUT_DIR, 'predictions_test_%s.csv' % arm)
        if not os.path.exists(per_arm):
            print('  no %s: run `--stage test --test-arm %s` first'
                  % (os.path.basename(per_arm), arm))
            return []
        src = per_arm
        print('  using %s' % os.path.basename(src))
    if not os.path.exists(src):
        print('  no outputs/predictions_test.csv: run `--stage test` first')
        return []
    d = pd.read_csv(src)
    d['DateTime'] = pd.to_datetime(d['DateTime'])
    d = d[np.isfinite(d.actual) & np.isfinite(d.predicted)]
    written = []
    pred_vs_obs(d, written)
    residual_hist(d, written)
    accuracy_by_station(d, written)
    timeseries_sample(d, written)
    error_vs_concentration(d, written)
    geocp_coverage(written)
    shap_importance(written)
    json.dump({'figures': [os.path.basename(p) for p in written], 'rows': int(len(d)), 'arm': arm},
              open(os.path.join(OUT, 'figures.json'), 'w'), indent=2)

    if arm:                                     # copy this arm's set to the top level as "latest"
        import shutil as _sh
        for p in written:
            try:
                _sh.copy2(p, os.path.join(base, os.path.basename(p)))
            except Exception:
                pass
    OUT = base
    arm_comparison(written)
    print('wrote %d figure(s)' % len(written))
    return written


def _pred_files():
    """{arm: predictions_test_<arm>.csv} for every arm that has been scored."""
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(data.OUT_DIR, 'predictions_test_*.csv'))):
        arm = os.path.basename(p)[len('predictions_test_'):-4]
        out[arm] = p
    return out


def arm_comparison(written):
    """Every scored arm's test-station scatter, plus R2/MAE bars.

    This is the figure the ablation actually needs: five arms, one table, one picture. It is written only
    when at least two arms have been scored.
    """
    plt = _mpl()
    files = _pred_files()
    if len(files) < 2:
        print('  arm_comparison.png         skipped (score another arm first)')
        return None

    import math
    order = [a for a in ('A', 'B', 'C', 'C+', 'TEXTURE') if a in files]
    order += [a for a in files if a not in order]
    stats, frames = {}, {}
    for a in order:
        d = pd.read_csv(files[a])
        d = d[np.isfinite(d['actual']) & np.isfinite(d['predicted'])]
        r2 = 1 - np.sum((d.actual - d.predicted) ** 2) / np.sum((d.actual - d.actual.mean()) ** 2)
        stats[a] = {'r2': float(r2), 'mae': float(np.mean(np.abs(d.actual - d.predicted)))}
        frames[a] = d

    ncols = min(3, len(order))
    nrows = math.ceil(len(order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.1 * nrows), squeeze=False)
    for ax in axes.ravel()[len(order):]:
        ax.set_axis_off()
    lim = [0, float(np.nanpercentile(frames[order[0]]['actual'], 99.5))]
    for ax, a in zip(axes.ravel(), order):
        d = frames[a]
        ax.scatter(d.actual, d.predicted, s=3, alpha=0.18, color='#2a6f77', edgecolors='none')
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title('%s  (%d feats)  R2 %.3f  MAE %.2f'
                     % (a, len(frames[a].columns) and int(_n_features(a)), stats[a]['r2'],
                        stats[a]['mae']), fontsize=9, loc='left')
        ax.set_xlabel('observed', fontsize=8); ax.set_ylabel('predicted', fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle('Test stations: every scored arm', fontsize=11, x=0.02, ha='left')
    _save(fig, 'arm_comparison.png', written)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10.4, 3.6))
    x = np.arange(len(order))
    cols = ['#2a6f77' if a == max(stats, key=lambda k: stats[k]['r2']) else '#95a5a6' for a in order]
    axes2[0].bar(x, [stats[a]['r2'] for a in order], color=cols)
    axes2[0].axhline(0, color='gray', lw=0.8)
    axes2[0].set_xticks(x); axes2[0].set_xticklabels(order)
    axes2[0].set_title('Test R2 by arm', fontsize=10, loc='left')
    axes2[1].bar(x, [stats[a]['mae'] for a in order], color=cols)
    axes2[1].set_xticks(x); axes2[1].set_xticklabels(order)
    axes2[1].set_title('Test MAE by arm (ug/m3)', fontsize=10, loc='left')
    _save(fig2, 'arm_comparison_bars.png', written)
    json.dump(stats, open(os.path.join(OUT, 'arm_comparison.json'), 'w'), indent=2)
    return stats


def _n_features(arm):
    """Feature count for an arm, without importing the whole ablation stack."""
    try:
        from training.ablation import ARM_FEATURES
        return len(ARM_FEATURES[arm])
    except Exception:
        return 0


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', default=None, help='use predictions_test_<arm>.csv')
    main(ap.parse_args().arm)
