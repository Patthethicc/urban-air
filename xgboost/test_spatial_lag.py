"""Regression test for compute_spatial_lag. Exits non-zero if any assertion fails.

The pd.DataFrame(...) call in compute_spatial_lag resolves a local named `weights` to pandas'
DataFrame.weights accessor, which collapsed every lag value to 1.0. These tests assert what that
bug broke: varying output, small row count, no self-inclusion, and no near-perfect correlation
with the target (which would mean the cell's own reading leaked into its own feature).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import prep_n_proc_pilot as prep  # noqa: E402
import spatial_cv  # noqa: E402

df = prep.df_clean
doc = spatial_cv.load_fold_map('fold_partitions.json')
parts = doc['partitions']
cent = spatial_cv.get_grid_centroids()
grid_ids = [int(g) for g in cent['Grid_ID']]
coords = cent[['cx', 'cy']].values

fails = []


def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    print(f'  [{status}] {name}' + (f' -- {detail}' if detail else ''))
    if not condition:
        fails.append(name)


def manual_lag(target, hours, neighbours, weights, obs):
    """Independent sum(w*x)/sum(w) over neighbours present, computed with plain Python loops."""
    out = []
    for h in hours:
        vals, ws = [], []
        for nid, w in zip(neighbours, weights):
            v = obs.loc[nid, h]
            if not pd.isna(v):
                vals.append(float(v))
                ws.append(float(w))
        out.append(sum(v * w for v, w in zip(vals, ws)) / sum(ws) if ws else np.nan)
    return np.array(out)


print('1. all-cells-allowed call (the case that used to collapse to 1.0)')
lag_all = prep.compute_spatial_lag(df, allowed_grid_ids=grid_ids, target_grid_ids=[226], k=4)
col = lag_all['pm25_spatial_lag'].dropna()
check('output has more than one distinct value', col.nunique() > 100,
      f'{col.nunique()} distinct')
check('output is not constant 1.0/log1p(1)', not np.allclose(col.unique(), 0.6931471805599453))
check('output is not constant 1.0 raw',
      not np.allclose(prep.compute_spatial_lag(df, allowed_grid_ids=grid_ids,
                                              target_grid_ids=[226], k=4, log_target=False)
                      ['pm25_spatial_lag'].dropna().unique(), 1.0))

print('\n2. row count equals the target cell\'s own (cell, hour) pairs')
expected = df[df['Grid_ID'] == 226][['Grid_ID', 'DateTime']].drop_duplicates().shape[0]
check('rows match distinct (Grid_ID, DateTime) of the target', len(lag_all) == expected,
      f'{len(lag_all)} vs {expected}')

print('\n3. matches an independent hand-rolled weighted mean')
obs = df.groupby(['Grid_ID', 'DateTime'])['Value'].mean().unstack('DateTime')
pos = grid_ids.index(226)
dist = np.sqrt(((coords - coords[pos]) ** 2).sum(axis=1))
order = [j for j in np.argsort(dist) if grid_ids[j] != 226]
nbr_ids = [grid_ids[j] for j in order[:4]]
nbr_w = 1.0 / dist[order[:4]]
hours = list(lag_all['DateTime'])
mine = manual_lag(226, hours, nbr_ids, nbr_w, obs)
theirs = lag_all.set_index('DateTime').loc[hours, 'pm25_spatial_lag'].values
m = ~np.isnan(mine) & ~np.isnan(theirs)
check('matches for every overlapping hour', np.allclose(np.log1p(mine[m]), theirs[m]),
      f'{m.sum()} hours compared, max diff '
      f'{np.nanmax(np.abs(np.log1p(mine[m]) - theirs[m])):.2e}')
check('NaN pattern matches (no neighbour reading -> NaN)',
      np.array_equal(np.isnan(mine), np.isnan(theirs)))

print('\n4. the target cell\'s own readings must not enter its own lag')
corrupt = df.copy()
corrupt.loc[corrupt['Grid_ID'] == 226, 'Value'] = 9999.0
dirty = prep.compute_spatial_lag(corrupt, allowed_grid_ids=grid_ids, target_grid_ids=[226], k=4)
check('lag identical after replacing the cell\'s own readings with 9999',
      lag_all['pm25_spatial_lag'].equals(dirty['pm25_spatial_lag']))

print('\n5. fold-aware coverage still behaves (held-out partition, train-only pool)')
train, test = parts['train'], parts['test']
lag_h = prep.compute_spatial_lag(df, allowed_grid_ids=train, target_grid_ids=test, k=4)
merged = df[df['Grid_ID'].isin(test)][['Grid_ID', 'DateTime', 'Value']].merge(
    lag_h, on=['Grid_ID', 'DateTime'], how='left')
check('held-out lag is mostly present', merged['pm25_spatial_lag'].notna().mean() > 0.9,
      f"{merged['pm25_spatial_lag'].notna().mean():.1%}")
corrupt2 = df.copy()
corrupt2.loc[corrupt2['Grid_ID'].isin(test), 'Value'] = 9999.0
dirty2 = prep.compute_spatial_lag(corrupt2, allowed_grid_ids=train, target_grid_ids=test, k=4)
check('held-out lag unaffected by its own readings',
      lag_h['pm25_spatial_lag'].equals(dirty2['pm25_spatial_lag']))

print('\n6. feature character: informative but not a copy of the target')
v = merged.dropna(subset=['pm25_spatial_lag'])
r = float(np.corrcoef(v['pm25_spatial_lag'], np.log1p(v['Value']))[0, 1])
check('correlation with the target is below 0.95', abs(r) < 0.95, f'r = {r:.3f}')
check('correlation with the target is above 0.1 (the feature carries signal)', abs(r) > 0.1,
      f'r = {r:.3f}')

print('\n' + ('ALL CHECKS PASSED' if not fails else f'FAILED: {fails}'))
sys.exit(1 if fails else 0)
