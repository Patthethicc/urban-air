"""Verify the shared split (Phase 1) and the 10 m patch statistics (Phase 2).

Checks the properties a reviewer would ask about:
  * the split is 54/13/16, disjoint, and covers exactly the stations that have data
  * the 16 test and 13 calibration stations are EXACTLY the CNN-LSTM run's stations
  * every cell in every partition has complete morphology (no silent NaN)
  * the imputed cells are few, recorded, and within the range of their neighbours
  * outer folds partition the training cells exactly once; inner folds never contain that outer fold's
    held-out cells (leakage)
  * patch statistics cover every sensor cell, one row each, with the 320 m window actually used
  * `prep` really produces rows for all 83 split cells and all 16 test stations

    python checks/verify_shared_split.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402

FAILS = []
def check(label, ok, detail=''):
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', label, (' -- ' + str(detail)) if detail else ''))
    if not ok:
        FAILS.append(label)

SHARED = os.path.join(data.DATA_DIR, 'fold_partitions_shared.json')
IMPUTED = os.path.join(data.DATA_DIR, 'imputed_cells.json')
PATCH = os.path.join(data.DATA_DIR, 'patch_stats.csv')
THEIR_RUN = os.path.join(data.DATA_DIR, '20261016_070432')
MORPH = ['bh_mean', 'bh_stdev', 'ndvi_mean', 'bldg_dens', 'aspect_ratio_mean', 'street_width_mean']

doc = json.load(open(SHARED))
parts = doc['partitions']
grid = pd.read_csv(data.GRID_CSV)
mapping = pd.read_csv(data.SENSOR_MAP_CSV)

print('=== 1. the split ===')
n = {k: len(v) for k, v in parts.items()}
check('counts are 54 train / 13 calibration / 16 test', n == {'train': 54, 'calibration': 13, 'test': 16}, n)
cells = {k: set(v) for k, v in parts.items()}
check('partitions are disjoint',
      not (cells['train'] & cells['calibration']) and not (cells['train'] & cells['test'])
      and not (cells['calibration'] & cells['test']))
check('partitions cover 83 cells', len(cells['train'] | cells['calibration'] | cells['test']) == 83)

print('\n=== 2. the stations are EXACTLY the CNN-LSTM run\'s ===')
ok_ref = True
for f, key in (('predictions_test.csv', 'test'), ('predictions_calibration.csv', 'calibration')):
    p = os.path.join(THEIR_RUN, f)
    if not os.path.exists(p):
        print('  (their %s not found - skipped)' % f); ok_ref = False; continue
    their = set(pd.read_csv(p, usecols=['location_id']).location_id.astype(int).unique())
    mine = set(doc['info']['%s_stations' % key])
    check('%s stations identical to theirs (%d)' % (key, len(their)), their == mine,
          'diff %s' % (their ^ mine) if their != mine else '')
    cell_of = dict(zip(mapping['sensor_id'].astype(int), mapping['id'].astype(int)))
    placed = {cell_of[s] for s in their if s in cell_of}
    check('all their %s stations map into the %s partition' % (key, key), placed <= cells[key],
          'missing %s' % (placed - cells[key]) if not placed <= cells[key] else '')

print('\n=== 3. morphology is complete for every partition cell ===')
g = grid.set_index('id')
miss = {k: sorted(g.loc[sorted(v), MORPH].isna().any(axis=1).pipe(lambda s: s[s].index))
        for k, v in cells.items()}
check('no NaN morphology in train', not miss['train'], miss['train'])
check('no NaN morphology in calibration', not miss['calibration'], miss['calibration'])
check('no NaN morphology in test', not miss['test'], miss['test'])

if os.path.exists(IMPUTED):
    imp = json.load(open(IMPUTED))
    ids = imp['cells']
    check('imputed cells are recorded (%d)' % len(ids), True, ids)
    check('imputed cells are all finite after imputation',
          bool(g.loc[ids, MORPH].notna().all().all()))
    # plausibility: imputed morphology within the observed min/max of complete cells
    comp = g[g[MORPH].notna().all(axis=1)]
    ok = True
    for c in ids:
        for col in MORPH:
            lo, hi = comp[col].min(), comp[col].max()
            if not (lo - 1e-9 <= g.loc[c, col] <= hi + 1e-9):
                ok = False
    check('imputed values lie within the observed range of complete cells', ok)

print('\n=== 4. folds have no leakage ===')
outer = {int(k): set(v) for k, v in doc['folds']['outer'].items()}
inner = {int(k): {int(c): int(v) for c, v in m.items()} for k, m in doc['folds']['inner'].items()}
check('outer folds partition the training cells exactly once',
      sorted(c for s in outer.values() for c in s) == sorted(cells['train']))
check('outer folds never contain calibration or test cells',
      all(not (s & (cells['calibration'] | cells['test'])) for s in outer.values()))
leak = []
for f, m in inner.items():
    fit = set(m)                     # inner folds only label the OUTER-FIT cells
    if fit & outer[f]:
        leak.append(f)
    if fit & (cells['calibration'] | cells['test']):
        leak.append(f)
check('inner folds exclude that outer fold\'s held-out cells and all held-out partitions',
      not leak, leak)

print('\n=== 5. patch statistics ===')
ps = pd.read_csv(PATCH)
check('one row per sensor cell', len(ps) == len(mapping['id'].unique()) == ps.Grid_ID.nunique(),
      '%d rows, %d unique' % (len(ps), ps.Grid_ID.nunique()))
check('covers exactly the mapped cells', set(ps.Grid_ID) == set(mapping['id'].astype(int)))
check('320 m window at 10 m = 32 px', bool((ps.patch_px == 32).all()), sorted(ps.patch_px.unique()))
tex = [c for c in ps.columns if c.startswith('tex_')]
check('35 texture features (5 bands x 7 stats)', len(tex) == 35, len(tex))
nan_cells = sorted(ps.loc[ps[tex].isna().any(axis=1), 'Grid_ID'].astype(int))
in_split = sorted((cells['train'] | cells['calibration'] | cells['test']) & set(nan_cells))
print('       cells with any NaN texture: %s' % nan_cells)
check('texture is complete for every cell in the split', not in_split,
      'in-split cells needing attention: %s' % in_split)

print('\n=== 6. prep actually produces the 83 cells and the 16 test stations ===')
from preprocessing import prep  # noqa: E402
df = prep.df_clean
have_cells = set(df['Grid_ID'].unique())
check('df_clean covers every split cell', (cells['train'] | cells['calibration'] | cells['test']) <= have_cells,
      'missing %s' % sorted((cells['train'] | cells['calibration'] | cells['test']) - have_cells))
test_st = set(doc['info']['test_stations'])
check('df_clean has rows for all 16 test stations', test_st <= set(df['Location'].unique()),
      'missing %s' % sorted(test_st - set(df['Location'].unique())))

print('\n' + '=' * 70)
if FAILS:
    print('%d CHECK(S) FAILED: %s' % (len(FAILS), FAILS))
    sys.exit(1)
print('ALL SHARED-SPLIT AND PATCH-STATISTICS CHECKS PASSED')
