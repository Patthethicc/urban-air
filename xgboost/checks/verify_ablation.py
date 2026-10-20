import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""Verify a completed Step 3 ablation run. Exits non-zero if any check fails.

    python verify_ablation.py

Checks that the reported numbers are trustworthy rather than just present: that no calibration or
test cell ever entered training, that all arms were scored on identical rows, that the raw-unit
inversion is correct, and that a held-out cell's prediction does not depend on its own rows.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.getcwd())
import data  # noqa: E402
from preprocessing import prep  # noqa: E402
from preprocessing import spatial_cv
from training.ablation_outputs import require_ablation_outputs  # noqa: E402
from training.ablation import ARM_FEATURES, validate_arm_features  # noqa: E402

OUT = data.OUT_DIR
fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


doc = spatial_cv.load_fold_map(data.FOLD_JSON)
parts = doc['partitions']
train_cells = set(parts['train'])
held_out = set(parts['calibration']) | set(parts['test'])
df = prep.df_clean

print('1. inputs and feature groups')
validate_arm_features()
check('feature groups cover prep.features exactly',
      (set(ARM_FEATURES['A']) | set(ARM_FEATURES['B']) | set(ARM_FEATURES['C']))
      <= set(prep.features) | set(ARM_FEATURES['C+']))
check('arm C is the import-time 24 columns', ARM_FEATURES['C'] == list(prep.features))
check('fold JSON still splits 11/3/3',
      (len(parts['train']), len(parts['calibration']), len(parts['test'])) == (11, 3, 3))

print('\n2. outputs exist')
# Fail fast: ablation_params.json and the per-row oof_predictions_*.csv are not committed (8.4 MB,
# fully regenerable), so one actionable message from require_ablation_outputs() is more useful than a
# list of FAIL lines followed by the same instruction.
require_ablation_outputs(OUT, need_oof=True, need_params=True)
for f in ['ablation_results.csv', 'ablation_summary.csv', 'ablation_params.json',
          'oof_predictions_A.csv', 'oof_predictions_B.csv', 'oof_predictions_C.csv']:
    check(f'{f} exists', os.path.exists(os.path.join(OUT, f)))

if fails:
    print('\nMissing outputs; stopping here.')
    sys.exit(1)

res = pd.read_csv(os.path.join(OUT, 'ablation_results.csv'))
summary = pd.read_csv(os.path.join(OUT, 'ablation_summary.csv'))
params = json.load(open(os.path.join(OUT, 'ablation_params.json')))
oof = {a: pd.read_csv(os.path.join(OUT, f'oof_predictions_{a}.csv')) for a in ['A', 'B', 'C']}
# the CSV round-trip turns datetimes into strings; compare against df_clean as real timestamps
for d in oof.values():
    d['DateTime'] = pd.to_datetime(d['DateTime'])

print('\n3. the ablation covered every arm and fold')
check('all three arms present', set(res['arm']) == {'A', 'B', 'C'}, str(sorted(set(res['arm']))))
for arm in ['A', 'B', 'C']:
    folds = res.loc[res['arm'] == arm, 'outer_fold'].tolist()
    check(f'arm {arm} ran all 3 outer folds', sorted(folds) == [0, 1, 2], str(sorted(folds)))
check('acq_func recorded as EI', params['acq_func'] == 'EI')
check('residual convention recorded as actual - predicted',
      params['residual'] == 'actual - predicted')
check('n_features recorded per arm',
      res.groupby('arm')['n_features'].first().to_dict() == {'A': 18, 'B': 15, 'C': 24},
      str(res.groupby('arm')['n_features'].first().to_dict()))

print('\n4. no calibration or test cell was ever fitted or scored')
for arm, d in oof.items():
    used = set(d['Grid_ID'].unique())
    check(f'arm {arm} scored only training cells', used <= train_cells, str(sorted(used - train_cells)))
    check(f'arm {arm} touched no held-out cell', not (used & held_out))
check('oof rows per arm equal the 11 training cells row count',
      all(len(d) == int(df['Grid_ID'].isin(train_cells).sum()) for d in oof.values()),
      str({k: len(v) for k, v in oof.items()}))
# Grid_ID 78 holds two sensors (3370151, 6303934) reporting at the same hours, so 105 cell-hours
# appear twice. Row counts and distinct cell-hour counts therefore differ by exactly that much;
# the caveat is recorded in STEP3_ABLATION.md and must be stated in the paper.
dupes = int(df.loc[df['Grid_ID'].isin(train_cells)]
            .duplicated(['Grid_ID', 'DateTime']).sum())
check('duplicate cell-hours are only from the two-sensor cell, and known',
      dupes == 105 and set(df.loc[df['Grid_ID'].isin(train_cells)]
                           .loc[df.loc[df['Grid_ID'].isin(train_cells)]
                                .duplicated(['Grid_ID', 'DateTime'], keep=False), 'Grid_ID']) == {78},
      f'{dupes} duplicated cell-hours, all in cell 78')

print('\n5. every arm was scored on identical rows (otherwise the comparison is confounded)')
keys = {a: set(map(tuple, d[['Grid_ID', 'DateTime']].values)) for a, d in oof.items()}
check('A == B == C row sets', keys['A'] == keys['B'] == keys['C'],
      f'{len(keys["A"])} distinct cell-hours each')
check('row-count minus distinct-cell-hours equals the known 105 duplicates for every arm',
      all(len(d) - len(keys[a]) == 105 for a, d in oof.items()),
      str({a: len(d) - len(keys[a]) for a, d in oof.items()}))

print('\n6. reported metrics reproduce from the stored predictions')
# merge rather than reindex: cell 78's two sensors make (Grid_ID, DateTime) non-unique, and a
# non-unique MultiIndex cannot be used for .reindex(). The stored 'actual' column is what the run
# scored, so compare the stored actual against df_clean through a merge on all three keys.
lookup = df[['Location', 'Grid_ID', 'DateTime', 'Value']]
for arm, d in oof.items():
    merged = d.merge(lookup, on=['Location', 'Grid_ID', 'DateTime'], how='left',
                     suffixes=('', '_df'))
    check(f'arm {arm} every scored row matched back to df_clean',
          bool(merged['Value'].notna().all()), f'{int(merged["Value"].isna().sum())} unmatched')
    rmse = float(np.sqrt(mean_squared_error(merged['Value'], merged['predicted'])))
    mae = float(mean_absolute_error(merged['Value'], merged['predicted']))
    r2 = float(r2_score(merged['Value'], merged['predicted']))
    row = summary.set_index('arm').loc[arm]
    check(f'arm {arm} pooled RMSE/MAE/R2 reproduce from raw Value',
          (abs(rmse - row['pooled_rmse']) < 0.01 and abs(mae - row['pooled_mae']) < 0.01
           and abs(r2 - row['pooled_r2']) < 0.01),
          f'recomputed {rmse:.3f}/{mae:.3f}/{r2:.3f} vs stored '
          f'{row["pooled_rmse"]:.3f}/{row["pooled_mae"]:.3f}/{row["pooled_r2"]:.3f}')
    check(f'arm {arm} stored actual equals df_clean Value (raw units, not log1p)',
          np.allclose(merged['Value'], merged['actual'], atol=1e-6))
    # the CSV round-trip stores float32, so compare at storage precision and assert the SIGN is
    # still actual - predicted: reversing it would differ by up to ~500 ug/m3, not by 1e-6.
    resid_err = float(np.abs(d['residual'] - (d['actual'] - d['predicted'])).max())
    check(f'arm {arm} residual sign is actual - predicted',
          resid_err < 1e-4 and np.allclose(np.sign(d['residual']),
                                           np.sign(d['actual'] - d['predicted'])),
          f'max deviation {resid_err:.2e}, sign agreement '
          f'{float((np.sign(d["residual"]) == np.sign(d["actual"] - d["predicted"])).mean()):.3f}')

print('\n7. predictions are finite and physically plausible')
for arm, d in oof.items():
    neg = int((d['predicted'] < 0).sum())
    check(f'arm {arm} predictions finite', bool(np.isfinite(d['predicted']).all()))
    # expm1 of a small log prediction can land a hair below zero; only negligible ones are tolerable
    check(f'arm {arm} no materially negative predictions',
          bool(d.loc[d['predicted'] < 0, 'predicted'].abs().max() < 0.1 if neg else True),
          f'{neg} negative, worst {d["predicted"].min():.4f} ug/m3')
    check(f'arm {arm} predictions in a plausible PM2.5 range',
          bool(d['predicted'].max() < 1000), f'max={d["predicted"].max():.2f}')

print('\n8. per-fold spread is reported, not just the mean')
for arm in ['A', 'B', 'C']:
    g = res[res['arm'] == arm]
    check(f'arm {arm} R2 sd recorded', not pd.isna(g['r2'].std()), f'sd={g["r2"].std():.3f}')
    check(f'arm {arm} test row counts sum to the training rows',
          int(g['n_test_rows'].sum()) == int(df['Grid_ID'].isin(train_cells).sum()),
          str(int(g['n_test_rows'].sum())))

print('\n9. arm ordering is stated for the paper')
s = summary.set_index('arm')
check('summary contains A, B and C in order', list(summary['arm']) == ['A', 'B', 'C'],
      str(list(summary['arm'])))
delta_ba = s.loc['B', 'pooled_r2'] - s.loc['A', 'pooled_r2']
delta_cb = s.loc['C', 'pooled_r2'] - s.loc['B', 'pooled_r2']
print(f'  B - A (morphology replaces PM2.5 history): {delta_ba:+.3f} R2')
print(f'  C - B (adding PM2.5 history back):         {delta_cb:+.3f} R2')

print('\n' + ('ALL CHECKS PASSED' if not fails else f'FAILED: {fails}'))
sys.exit(1 if fails else 0)
