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

STAGES = ('report', 'partitions', 'ablation', 'geocp', 'save', 'infer', 'checks', 'all')


def stage_report():
    from reporting import dataset_report
    r = dataset_report.dataset_report(write=True)
    print(f"  rows {r['rows_kept']:,} kept of {r['raw_rows']:,}; "
          f"{r['cells_with_sensor']} cells with a sensor")


def stage_partitions():
    from training import partitions
    partitions.main()


def stage_ablation(n_trials=20, arms=('A', 'B', 'C')):
    from training.ablation import run
    run(arms=list(arms), n_trials=n_trials, out_dir=data.OUT_DIR)


def stage_geocp(alpha=0.10, h_mode='thesis-grid'):
    from uncertainty import geocp
    geocp.run(alphas=(alpha,), h_mode=h_mode)


def _fit_production_model():
    """Fit the arm-C model on the proper-training cells, using the best-generalising outer fold.

    Shared by the save and infer stages so both use exactly the same model. Returns
    (model, features, info) where info carries what the artifact needs.
    """
    import json
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    from preprocessing import prep
    from preprocessing import spatial_cv
    from training.ablation import ARM_FEATURES

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
    rows = results[results['arm'] == 'C'].set_index('outer_fold')
    best_fold = int(rows['r2'].idxmax())
    params = json.loads(rows.loc[best_fold, 'best_params'])
    n_est = int(rows.loc[best_fold, 'n_estimators'])

    features = list(prep.features)
    X = df_train[features].to_numpy(dtype=float)
    y = np.log1p(df_train['Value'].to_numpy())
    model = xgb.XGBRegressor(n_estimators=n_est, random_state=42, tree_method='hist', n_jobs=-1,
                             **params)
    model.fit(X, y)
    print(f"  arm C, {len(features)} features, fold {best_fold}'s hyperparameters, n_estimators={n_est}")
    print(f"  fitted on {len(df_train):,} rows from {df_train['Grid_ID'].nunique()} training cells")
    info = {'arm': 'C', 'best_fold': best_fold, 'params': params, 'n_estimators': n_est,
            'partitions': partitions, 'fold_doc': fold_doc, 'shape': X.shape}
    return model, features, info


def stage_save(run_id=None):
    from training.artifacts import save_model_artifact
    model, features, info = _fit_production_model()
    out = save_model_artifact(
        model, features, out_dir=data.out('artifacts', run_id or 'latest'),
        arm=info['arm'], best_fold=info['best_fold'], params=info['params'],
        n_estimators=info['n_estimators'], partitions=info['partitions'],
        fold_doc=info['fold_doc'], training_shape=info['shape'])
    print(f'  artifact written to {out}')
    return out


def stage_infer(date=None, timestamps=None, out_dir=None):
    """City-wide surfaces for the four snapshot hours of each date (thesis Section 4.6.2)."""
    import pandas as pd
    from inference.cell_inference import infer_timestamp
    from output import surfaces

    model, features, info = _fit_production_model()
    hours = [8, 12, 17, 21]
    if timestamps:
        stamps = [pd.Timestamp(t) for t in timestamps]
    else:
        dates = [date] if date else ['2026-04-11']
        stamps = [pd.Timestamp(d).normalize() + pd.Timedelta(hours=h) for d in dates for h in hours]

    written = []
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
        cells = surfaces.geocp_intervals(model, cells)
        written += surfaces.write_snapshot(cells, ts, out_dir=out_dir)
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
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--h-mode', default='thesis-grid',
                    choices=('thesis-grid', 'median-distance'))
    ap.add_argument('--date', default=None, help='YYYY-MM-DD for the infer stage')
    ap.add_argument('--timestamp', nargs='+', default=None,
                    help='explicit timestamps for infer, e.g. "2026-04-11 08:00"')
    ap.add_argument('--run-id', default=None, help='artifact subfolder name (save stage)')
    ap.add_argument('--out-dir', default=None, help='output folder for the infer stage')
    args = ap.parse_args()

    data.ensure_dirs()
    print(f'stage: {args.stage}')
    if args.stage == 'report':
        stage_report()
    elif args.stage == 'partitions':
        stage_partitions()
    elif args.stage == 'ablation':
        stage_ablation(args.n_trials)
    elif args.stage == 'geocp':
        stage_geocp(args.alpha, args.h_mode)
    elif args.stage == 'save':
        stage_save(args.run_id)
    elif args.stage == 'infer':
        stage_infer(args.date, args.timestamp, args.out_dir)
    elif args.stage == 'checks':
        failed = stage_checks()
        if failed:
            raise SystemExit(f'checks failed: {failed}')
    elif args.stage == 'all':
        stage_report()
        stage_partitions()
        stage_ablation(args.n_trials)
        stage_geocp(args.alpha, args.h_mode)
        stage_save(args.run_id)
    print('done')


if __name__ == '__main__':
    main()
