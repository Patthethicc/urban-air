"""Staged entry point for the XGBoost branch, mirroring the CNN-LSTM branch's run_pipeline.py.

    python run_pipeline.py --stage report      # Deliverable 1 dataset report
    python run_pipeline.py --stage partitions  # rebuild + validate the 11/3/3 split
    python run_pipeline.py --stage ablation    # the three-way ablation          (~7 min)
    python run_pipeline.py --stage geocp       # conformal intervals and coverage
    python run_pipeline.py --stage save        # Step 7: model artifact
    python run_pipeline.py --stage infer --date 2026-04-11   # Step 8: city-wide surfaces
    python run_pipeline.py --stage checks      # run every verification script
    python run_pipeline.py --stage all         # report -> partitions -> ablation -> geocp -> save

Stages read what the previous one wrote. `ablation` must precede `geocp`, which loads the tuned
parameters from outputs/ablation_results.csv, and `ablation` or `save` must precede `infer`, which
needs a fitted model.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data  # noqa: E402

STAGES = ('report', 'partitions', 'split', 'patch', 'ablation', 'test', 'geocp',
          'save', 'infer', 'figures', 'checks', 'all')


def stage_report():
    from reporting import dataset_report
    r = dataset_report.dataset_report(write=True)
    print(f"  rows {r['rows_kept']:,} kept of {r['raw_rows']:,}; "
          f"{r['cells_with_sensor']} cells with a sensor")


def stage_partitions(n_calibration=None, n_test=None):
    from training import partitions
    partitions.main(n_calibration=n_calibration, n_test=n_test)


def stage_split():
    """Write data/fold_partitions_shared.json (the CNN-LSTM run's 54/13/16 station split)."""
    from training import shared_split
    shared_split.main()


def stage_patch():
    """Write data/patch_stats.csv (10 m patch statistics for the TEXTURE arm)."""
    from preprocessing import patch_stats
    patch_stats.main()


def stage_test(arm='C'):
    """Score the production model on the reserved test stations."""
    from training import final_test
    return final_test.run(arm)


def stage_figures(arm=None):
    """Write the report figures to outputs/figures/.

    `arm` selects predictions_test_<arm>.csv, so each arm gets its own figure set; None uses the most
    recent test run (outputs/predictions_test.csv).
    """
    from reporting import figures
    return figures.main(arm)


def stage_ablation(n_trials=20, arms=('A', 'B', 'C'), smoke=False):
    from training.ablation import run
    run(arms=list(arms), n_trials=n_trials, out_dir=data.OUT_DIR, smoke=smoke)


def stage_geocp(alpha=0.10, h_mode='thesis-grid', max_cal=None, max_eval=None, profile='default'):
    """profile='shared' reproduces the CNN-LSTM run's GeoCP search (kernels x bandwidths, their
    correction setting) so the coverage numbers are directly comparable; 'default' keeps this
    branch's original single-kernel, finer-grid search."""
    if profile == 'shared':
        from uncertainty import geocp_shared
        return geocp_shared.run(max_cal=max_cal or 6000, max_eval=max_eval or 20000)
    from uncertainty import geocp
    geocp.run(alphas=(alpha,), h_mode=h_mode, max_cal=max_cal, max_eval=max_eval)


def best_arm():
    """The arm with the highest pooled R2 in ablation_summary.csv -- the one to deploy.

    This exists because `--stage save` and `--stage infer` used to hardcode arm C. After the five-arm
    run, C was NOT the best (A scored 0.666 vs C 0.655 on the test stations), so the artifact and every
    city-wide surface were being produced by the second-best model.
    """
    import pandas as pd
    path = os.path.join(data.OUT_DIR, 'ablation_summary.csv')
    if os.path.exists(path):
        s = pd.read_csv(path)
        if len(s):
            return str(s.sort_values('pooled_r2', ascending=False)['arm'].iloc[0])
    return 'C'


def _fit_production_model(arm=None):
    """Fit the production model on the proper-training cells, using the best-generalising outer fold.

    Shared by the save and infer stages so both use exactly the same model. Returns
    (model, features, info) where info carries what the artifact needs.
    """
    import json
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    from preprocessing import prep
    from preprocessing import spatial_cv
    from training.ablation import ARM_FEATURES, build_design_matrix

    fold_doc = spatial_cv.load_fold_map()
    partitions = fold_doc['partitions']
    df = prep.df_clean
    df_train = df[df['Grid_ID'].isin(partitions['train'])].reset_index(drop=True)

    results_path = data.ABLATION_RESULTS
    if not os.path.exists(results_path):
        raise SystemExit(
            f'{results_path} is missing. Run `python run_pipeline.py --stage ablation` first; the '
            f'production model takes its hyperparameters from that run.')
    results = pd.read_csv(results_path)
    arm = arm or best_arm()
    rows = results[results['arm'] == arm]
    if rows.empty:
        print('  WARNING: no ablation rows for arm %s; falling back to C' % arm)
        arm = 'C'
        rows = results[results['arm'] == 'C']
    rows = rows.set_index('outer_fold')
    best_fold = int(rows['r2'].idxmax())
    params = json.loads(rows.loc[best_fold, 'best_params'])
    n_est = int(rows.loc[best_fold, 'n_estimators'])

    features = list(ARM_FEATURES[arm])
    if arm == 'C+':
        # No held-out cells remain at deployment time, so every training cell may inform the lag.
        train_cells = [int(c) for c in df_train['Grid_ID'].unique()]
        print('  computing spatial lag for the production fit (k=5, %d cells) ...' % len(train_cells))
        lag = prep.compute_spatial_lag(df_train, allowed_grid_ids=train_cells,
                                       target_grid_ids=train_cells, k=5)
        X = build_design_matrix(df_train, arm, spatial_lag=lag)
    else:
        X = build_design_matrix(df_train, arm)
    y = np.log1p(df_train['Value'].to_numpy())
    model = xgb.XGBRegressor(n_estimators=n_est, random_state=42, tree_method='hist', n_jobs=-1,
                             **params)
    model.fit(X, y)
    print(f"  deploying arm {arm}, {len(features)} features, fold {best_fold}'s hyperparameters, "
          f"n_estimators={n_est}")
    print(f"  fitted on {len(df_train):,} rows from {df_train['Grid_ID'].nunique()} training cells")
    info = {'arm': arm, 'best_fold': best_fold, 'params': params, 'n_estimators': n_est,
            'partitions': partitions, 'fold_doc': fold_doc, 'shape': X.shape}
    return model, features, info


def stage_save(run_id=None, arm=None):
    from training.artifacts import save_model_artifact
    model, features, info = _fit_production_model(arm)
    out = save_model_artifact(
        model, features, out_dir=data.out('artifacts', run_id or 'latest'),
        arm=info['arm'], best_fold=info['best_fold'], params=info['params'],
        n_estimators=info['n_estimators'], partitions=info['partitions'],
        fold_doc=info['fold_doc'], training_shape=info['shape'])
    print(f'  artifact written to {out}')
    return out


def stage_infer(date=None, timestamps=None, out_dir=None, arm=None):
    """City-wide surfaces for the four snapshot hours of each date (thesis Section 4.6.2)."""
    import pandas as pd
    from inference.cell_inference import infer_timestamp
    from maps import surfaces

    model, features, info = _fit_production_model(arm)
    hours = [8, 12, 17, 21]
    if timestamps:
        stamps = [pd.Timestamp(t) for t in timestamps]
    else:
        dates = [date] if date else ['2026-04-11']
        stamps = [pd.Timestamp(d).normalize() + pd.Timedelta(hours=h) for d in dates for h in hours]

    written = []
    frames = {}
    for ts in stamps:
        try:
            cells, _ = infer_timestamp(model, ts, feature_names=features)
        except ValueError as exc:
            print(f'  {ts}: skipped ({exc})')
            continue
        usable = int(cells['prediction_usable'].sum())
        print(f'  {ts}: {usable}/{len(cells)} cells usable, '
              f'mean stand-in coverage {cells["standin_coverage"].mean():.2f}')
        if not usable:
            continue
        # GeoCP intervals on the same cells, calibrated on the cells the model never trained on
        cells = surfaces.geocp_intervals(model, cells, feature_names=features)
        frames[ts] = cells
        written += surfaces.write_snapshot(cells, ts, out_dir=out_dir)
    # ---- one shared colour scale across the day, then the panel and the daily-mean map
    if frames:
        sdir = out_dir or data.out("surfaces")
        day = sorted({pd.Timestamp(t).strftime("%Y%m%d") for t in frames})[0]
        lims = surfaces.suggest_limits(frames)
        written.append(surfaces.render_panel(
            frames, os.path.join(sdir, day, f"panel_{day}.png"),
            vmin=lims[0], vmax=lims[1],
            title=f"PM2.5 prediction, {day[:4]}-{day[4:6]}-{day[6:]} (continuous scale, "
                  f"shared {lims[0]:.1f}-{lims[1]:.1f} ug/m3)"))
        _mean = surfaces.render_daily_mean(frames, os.path.join(sdir, day), day,
                                          vmin=lims[0], vmax=lims[1])
        if _mean:
            written.append(_mean)
        print(f"  shared colour scale {lims[0]:.1f}-{lims[1]:.1f} ug/m3; panel + daily mean written")
    if written:
        print(f'  wrote {len(written)} file(s) under {out_dir or data.out("surfaces")}')
    return written


def stage_checks():
    import glob
    shims = sorted(glob.glob(os.path.join(ROOT, 'checks', '*.py')))
    shims = [p for p in shims if not p.endswith('__init__.py')]
    failed = []
    for p in shims:
        name = os.path.basename(p)
        r = subprocess.run([sys.executable, p], cwd=ROOT, capture_output=True, text=True)
        status = 'ok' if r.returncode == 0 else f'FAILED (exit {r.returncode})'
        print(f'  {name:32s} {status}')
        if r.returncode != 0:
            failed.append(name)
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=STAGES)
    ap.add_argument('--n-trials', type=int, default=20)
    ap.add_argument('--arms', nargs='+', default=None,
                    choices=['A', 'B', 'C', 'C+', 'TEXTURE'],
                    help='ablation arms to run (default: A B C)')
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--h-mode', default='thesis-grid',
                    choices=('thesis-grid', 'median-distance'))
    ap.add_argument('--date', default=None, help='YYYY-MM-DD for the infer stage')
    ap.add_argument('--timestamp', nargs='+', default=None,
                    help='explicit timestamps for infer, e.g. "2026-04-11 08:00"')
    ap.add_argument('--run-id', default=None, help='artifact subfolder name (save stage)')
    ap.add_argument('--out-dir', default=None, help='output folder for the infer stage')
    ap.add_argument('--geocp-profile', default='default', choices=['default', 'shared'],
                    help='shared = the CNN-LSTM run\'s kernel x bandwidth search')
    ap.add_argument('--test-arm', default='C', choices=['A', 'B', 'C', 'C+', 'TEXTURE'],
                    help='which arm to score on the test stations')
    ap.add_argument('--save-arm', default=None, choices=['A', 'B', 'C', 'C+', 'TEXTURE'],
                    help='save/infer: arm to deploy (default: best pooled R2 in ablation_summary)')
    ap.add_argument('--figures-arm', default=None, choices=['A', 'B', 'C', 'C+', 'TEXTURE'],
                    help='figures: use predictions_test_<arm>.csv instead of the latest test run')
    ap.add_argument('--n-calibration', type=int, default=None,
                    help='calibration cells (default: max(3, round(0.15 * cells)))')
    ap.add_argument('--n-test', type=int, default=None,
                    help='test cells (default: max(3, round(0.15 * cells)))')
    ap.add_argument('--no-infer', action='store_true',
                    help='--stage all: skip the city-wide surfaces (they are the slow tail)')
    ap.add_argument('--smoke', action='store_true',
                    help='ablation: 1 trial, tiny forest, subsampled rows -- checks the chain in minutes')
    ap.add_argument('--max-cal', type=int, default=None,
                    help='geocp: cap calibration records (seeded subsample; default: use all)')
    ap.add_argument('--max-eval', type=int, default=None,
                    help='geocp: cap test rows used for evaluation')
    args = ap.parse_args()

    data.ensure_dirs()
    # Never let a .env silently change which split a run uses: print the resolved configuration.
    print(f'stage: {args.stage}')
    print(f'  dataset={data.DATASET}  folds={os.path.basename(data.FOLD_JSON)}  '
          f'master={os.path.basename(data.MASTER_CSV)}  grid={os.path.basename(data.GRID_CSV)}')
    if args.stage == 'report':
        stage_report()
    elif args.stage == 'partitions':
        stage_partitions(args.n_calibration, args.n_test)
    elif args.stage == 'split':
        stage_split()
    elif args.stage == 'patch':
        stage_patch()
    elif args.stage == 'ablation':
        stage_ablation(args.n_trials, arms=args.arms or ('A', 'B', 'C'), smoke=args.smoke)
    elif args.stage == 'test':
        stage_test(args.test_arm)
    elif args.stage == 'figures':
        stage_figures(args.figures_arm)
    elif args.stage == 'geocp':
        stage_geocp(args.alpha, args.h_mode, args.max_cal, args.max_eval,
                    args.geocp_profile)
    elif args.stage == 'save':
        stage_save(args.run_id, args.save_arm)
    elif args.stage == 'infer':
        stage_infer(args.date, args.timestamp, args.out_dir, args.save_arm)
    elif args.stage == 'checks':
        failed = stage_checks()
        if failed:
            raise SystemExit(f'checks failed: {failed}')
    elif args.stage == 'all':
        # Split and patch statistics are cheap and idempotent, so they always run first; then the
        # experiment, the test read, intervals, the artifact, the maps and the figures.
        stage_split()
        stage_patch()
        stage_report()
        stage_ablation(args.n_trials, arms=args.arms or ('A', 'B', 'C'), smoke=args.smoke)
        stage_test(args.test_arm)
        stage_geocp(args.alpha, args.h_mode, args.max_cal, args.max_eval,
                    args.geocp_profile)
        stage_save(args.run_id, args.save_arm)
        if not args.no_infer:
            stage_infer(args.date, args.timestamp, args.out_dir, args.save_arm)
        stage_figures(args.figures_arm)
    print('done')


if __name__ == '__main__':
    main()
