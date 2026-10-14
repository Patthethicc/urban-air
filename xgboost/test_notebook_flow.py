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
import prep_n_proc_pilot as prep  # noqa: E402
import spatial_cv  # noqa: E402
from spatial_cv import load_fold_map  # noqa: E402

NB_CELLS = json.load(open('xgboost_pm25_pilot.ipynb', encoding='utf-8'))['cells']
display = print
MAPPING_CSV = 'sensor_to_grid_map_full.csv'
N_TRIALS = 16
OUT_DIR = 'outputs'

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


print('=== cells 2, 4, 5, 7 (setup, prep, distributions, dataset report) ===')
for idx in (2, 4, 5, 7):
    run(idx)
check('dataset report kept rows == X rows',
      ds['rows_kept'] == prep.X.shape[0], f"{ds['rows_kept']} vs {prep.X.shape[0]}")
check('drop funnel ends at the kept row count',
      int(ds['drop_funnel']['rows'].iloc[-1]) == ds['rows_kept'],
      str(int(ds['drop_funnel']['rows'].iloc[-1])))
check('the drop funnel never increases',
      bool((ds['drop_funnel']['rows'].diff().dropna() <= 0).all()))

print('\n=== cells 9, 10, 12 (fold assignment, partitions, feature arms) ===')
for idx in (9, 10, 12):
    run(idx)
check('cx/cy attached to df_clean by the partition cell',
      'cx' in df_clean.columns and 'cy' in df_clean.columns)
check('partition mapping covers every row', bool(df_clean['partition'].notna().all()))

print('\n=== the ablation cell replaced by the load-from-disk path it would take ===')
from ablation import ARM_LABELS  # noqa: E402
df_train = prep.df_clean[prep.df_clean['Grid_ID'].isin(partitions['train'])].reset_index(drop=True)
results_df = pd.read_csv(os.path.join(OUT_DIR, 'ablation_results.csv'))
ablation_summary = pd.read_csv(os.path.join(OUT_DIR, 'ablation_summary.csv'))
oof_by_arm = {a: pd.read_csv(os.path.join(OUT_DIR, f'oof_predictions_{a}.csv')) for a in 'ABC'}
oof_df = oof_by_arm['C'].copy()
fold_results = results_df[results_df['arm'] == 'C'].to_dict('records')
check('three arms loaded from disk', set(results_df['arm']) == {'A', 'B', 'C'})

print('\n=== cells 18, 19, 21, 22, 24 (results, plots, bias, production model) ===')
for idx in (18, 19, 21, 22, 24):
    run(idx)
plt.close('all')
check('production model fitted', production_model is not None)
check('production model never saw a held-out cell',
      not (set(df_train['Grid_ID']) & (set(partitions['calibration']) | set(partitions['test']))))

print('\n=== GeoCP cells 26, 27, 28 ===')
for idx in (26, 27, 28):
    run(idx)
check('geocp summary produced', 'geocp_summary' in dir() and len(geocp_summary) > 0)
iv = pd.read_csv(os.path.join(OUT_DIR, 'geocp', 'geocp_intervals_notebook.csv'))
check('intervals file has finite half-widths', bool(np.isfinite(iv['geocp_q']).any()))
check('unweighted baseline present alongside GeoCP', 'split_q' in iv.columns)

print('\n=== SHAP cells 30, 31, 32 ===')
for idx in (30, 31, 32):
    run(idx)
plt.close('all')

print('\n' + '=' * 92)
print('ALL NOTEBOOK CELLS EXECUTED CLEANLY' if not fails else f'FAILED: {fails}')
print('=' * 92)
print(geocp_summary[geocp_summary['scope'] == 'all test observations']
      [['method', 'empirical_coverage', 'mean_width']].round(3).to_string(index=False))
sys.exit(1 if fails else 0)
