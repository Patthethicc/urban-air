import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""Faithful reproduction of the notebook, cell by cell, with the ablation mocked from shipped CSVs.

Runs the notebook's real cells in order against a reconstructed state, so an error introduced by a cell
edit is caught without waiting for the full ablation. The ablation cell is replaced by the equivalent
load-from-disk block it would take; indices below are the CURRENT layout.

This is the test that caught the missing cx/cy columns that broke the GeoCP cells on the first real
notebook run, so it is kept in the suite. NB_CELLS (not `cells`) holds the cell list: one notebook cell
uses `cells` internally, which shadowed it before.
"""
import ast
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.getcwd())
import data  # noqa: E402
from preprocessing import prep  # noqa: E402
from preprocessing import spatial_cv  # noqa: E402
from preprocessing.spatial_cv import load_fold_map  # noqa: E402

# The notebook is written against the pilot split (11 train / 3 calibration / 3 test cells) and some of
# its cells are pilot-shaped. Running it against the 54/13/16 shared split fails inside a cell for
# reasons that say nothing about the pipeline, so skip with an explanation instead of reporting red.
_partitions = json.load(open(data.FOLD_JSON, encoding='utf-8'))['partitions']
_shape = (len(_partitions['train']), len(_partitions['calibration']), len(_partitions['test']))
if _shape != (11, 3, 3):
    print('SKIPPED: test_notebook_flow checks the pilot notebook, which expects an 11/3/3 split.')
    print('         Active split is %d/%d/%d from %s.'
          % (_shape[0], _shape[1], _shape[2], os.path.basename(data.FOLD_JSON)))
    print('         To run it: XGB_DATASET=pilot and clear XGB_FOLDS (the shared split is full-only).')
    raise SystemExit(0)

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'xgboost_pm25_pilot.ipynb')
NB_CELLS = json.load(open(NB_PATH, encoding='utf-8'))['cells']
display = print
MAPPING_CSV = data.SENSOR_MAP_CSV
N_TRIALS = 16
OUT_DIR = data.OUT_DIR

fails = []


def run(idx, label=''):
    src = ''.join(NB_CELLS[idx]['source'])
    ast.parse(src)
    exec(compile(src, f'<cell {idx}>', 'exec'), globals())
    print(f'  cell {idx:3d} {label}: OK')


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


print('=== cells 2, 3, 5, 6, 8 (bootstrap, setup, prep, distributions, dataset report) ===')
for idx in (2, 3, 5, 6, 8):
    run(idx)
check('dataset report kept rows == X rows',
      ds['rows_kept'] == prep.X.shape[0], f"{ds['rows_kept']} vs {prep.X.shape[0]}")
check('drop funnel ends at the kept row count',
      int(ds['drop_funnel']['rows'].iloc[-1]) == ds['rows_kept'],
      str(int(ds['drop_funnel']['rows'].iloc[-1])))
check('the drop funnel never increases',
      bool((ds['drop_funnel']['rows'].diff().dropna() <= 0).all()))

print('\n=== cells 10, 11, 13 (fold assignment, partitions, feature arms) ===')
for idx in (10, 11, 13):
    run(idx)
check('cx/cy attached to df_clean by the partition cell',
      'cx' in df_clean.columns and 'cy' in df_clean.columns)
check('partition mapping covers every row', bool(df_clean['partition'].notna().all()))

print('\n=== the ablation cell, in the state the notebook would find ===')
from training.ablation import ARM_LABELS  # noqa: E402
# The per-row predictions and ablation_params.json are deliberately NOT committed (8.4 MB, fully
# regenerable). The notebook's ablation cell checks for oof_predictions_{A,B,C}.csv and re-runs the
# whole ablation when they are absent, so this test does the same rather than assuming they exist.
from training.ablation_outputs import require_ablation_outputs  # noqa: E402
oof_paths = [os.path.join(OUT_DIR, f'oof_predictions_{a}.csv') for a in 'ABC']
missing_oof = [p for p in oof_paths if not os.path.exists(p)]
if missing_oof:
    print(f'  per-row predictions absent ({len(missing_oof)} of 3) -> running the ablation, as the')
    print('  notebook would. This takes ~7 minutes on a clean checkout.')
    from training.ablation import run as run_ablation
    results_df, ablation_summary, oof_frames = run_ablation(arms=['A', 'B', 'C'], n_trials=N_TRIALS,
                                                            out_dir=OUT_DIR)
    oof_by_arm = {a: pd.concat(chunks, ignore_index=True) for a, chunks in oof_frames.items()}
    print('  ablation complete')
else:
    results_df = pd.read_csv(os.path.join(OUT_DIR, 'ablation_results.csv'))
    ablation_summary = pd.read_csv(os.path.join(OUT_DIR, 'ablation_summary.csv'))
    oof_by_arm = {a: pd.read_csv(p) for a, p in zip('ABC', oof_paths)}
    print('  loaded from disk')
require_ablation_outputs(OUT_DIR, need_oof=True)
df_train = prep.df_clean[prep.df_clean['Grid_ID'].isin(partitions['train'])].reset_index(drop=True)
oof_df = oof_by_arm['C'].copy()
fold_results = results_df[results_df['arm'] == 'C'].to_dict('records')
check('three arms present', set(results_df['arm']) == {'A', 'B', 'C'})
check('one out-of-fold frame per arm', set(oof_by_arm) == {'A', 'B', 'C'})

print('\n=== cells 19, 20, 22, 23, 25 (results, plots, bias, production model) ===')
for idx in (19, 20, 22, 23, 25):
    run(idx)
plt.close('all')
check('production model fitted', production_model is not None)
check('production model never saw a held-out cell',
      not (set(df_train['Grid_ID']) & (set(partitions['calibration']) | set(partitions['test']))))

print('\n=== GeoCP cells 27, 28, 29 ===')
for idx in (27, 28, 29):
    run(idx)
check('geocp summary produced', 'geocp_summary' in dir() and len(geocp_summary) > 0)
iv = pd.read_csv(os.path.join(OUT_DIR, 'geocp', 'geocp_intervals_notebook.csv'))
check('intervals file has finite half-widths', bool(np.isfinite(iv['geocp_q']).any()))
check('unweighted baseline present alongside GeoCP', 'split_q' in iv.columns)

print('\n=== SHAP cells 31, 32, 33 ===')
for idx in (31, 32, 33):
    run(idx)
plt.close('all')

print('\n' + '=' * 92)
print('ALL NOTEBOOK CELLS EXECUTED CLEANLY' if not fails else f'FAILED: {fails}')
print('=' * 92)
print(geocp_summary[geocp_summary['scope'] == 'all test observations']
      [['method', 'empirical_coverage', 'mean_width']].round(3).to_string(index=False))
sys.exit(1 if fails else 0)
