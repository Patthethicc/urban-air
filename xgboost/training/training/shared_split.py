"""Phase 1: adopt the CNN-LSTM run's exact station split, so the two branches are comparable.

The CNN-LSTM run in `data/20261016_070432` partitioned 83 usable stations as
16 test / 13 calibration / 54 train. Comparing our numbers to theirs is meaningless unless the SAME
stations land in the SAME partition, so this takes their station ids verbatim (read from
`predictions_test.csv` and `predictions_calibration.csv`, which list every scored station) and builds a
matching fold document in the format `training.ablation` already expects.

    python -m training.shared_split

One known gap: station 9000015 (Susano Road, Quezon City) is in their TEST set, but
`master_raster_v2.tif` has no building data in its cell, so its morphology would be NaN and `prep`'s
dropna would silently delete the station -- losing 1 of 16 test stations. Its morphology is therefore
imputed by inverse-distance weighting from the k nearest complete cells, the filled cell is recorded in
`data/imputed_cells.json`, and the station is listed in the split's `info` so the write-up can report
test results with and without it. Their 10 m raster covers that cell; ours does not.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import data
from preprocessing import spatial_cv

OUT_JSON = os.path.join(data.DATA_DIR, 'fold_partitions_shared.json')
IMPUTED_JSON = os.path.join(data.DATA_DIR, 'imputed_cells.json')

# verbatim from data/20261016_070432/predictions_{test,calibration}.csv
TEST_STATIONS = [4526988, 4526989, 4797638, 4797658, 6303937, 9000002, 9000009, 9000010,
                 9000015, 9000018, 9000019, 9000022, 9000028, 9000030, 9000034, 9000038]
CALIBRATION_STATIONS = [3990425, 3990426, 4795499, 4797643, 4797650, 4797661, 9000001,
                        9000008, 9000013, 9000014, 9000016, 9000027, 9000039]

MORPH_COLS = ['bh_mean', 'bh_stdev', 'ndvi_mean', 'bldg_dens', 'aspect_ratio_mean',
              'street_width_mean']
KNN_IMPUTE = 4


def impute_missing_morphology(grid, cells_needed, k=KNN_IMPUTE):
    """IDW-fill morphology for needed cells that are NaN; leave every other cell untouched.

    `cells_needed` is the set of grid ids that hold a station we must keep. Only a cell in that set
    AND with missing morphology is modified. Returns (grid, list_of_imputed_cell_ids).
    """
    complete = grid[grid[MORPH_COLS].notna().all(axis=1)]
    targets = grid[grid['id'].isin(cells_needed) & grid[MORPH_COLS].isna().any(axis=1)]
    if targets.empty or complete.empty:
        return grid, []

    tree = cKDTree(np.column_stack([complete['left'].to_numpy(float) + 250.0,
                                    complete['bottom'].to_numpy(float) + 250.0]))
    grid = grid.copy()
    imputed = []
    for idx, cell in targets.iterrows():
        q = np.array([cell['left'] + 250.0, cell['bottom'] + 250.0])
        d, j = tree.query(q, k=min(k, len(complete)))
        d, j = np.atleast_1d(d), np.atleast_1d(j)
        w = 1.0 / np.maximum(d, 1.0)
        w = w / w.sum()
        for col in MORPH_COLS:
            if pd.isna(grid.at[idx, col]):
                grid.at[idx, col] = float(np.dot(w, complete[col].to_numpy(float)[j]))
        imputed.append(int(cell['id']))
    return grid, imputed


def main():
    from preprocessing import prep  # noqa: F401  (import builds the sensor->cell map)
    mapping = pd.read_csv(data.SENSOR_MAP_CSV)
    print('sensor -> cell map: %d stations, %d cells' % (len(mapping), mapping['id'].nunique()))

    cell_of = dict(zip(mapping['sensor_id'].astype(int), mapping['id'].astype(int)))
    wanted = TEST_STATIONS + CALIBRATION_STATIONS
    missing = [s for s in wanted if s not in cell_of]
    if missing:
        print('  WARNING: stations in their split we cannot place: %s' % missing)

    # ---- fill morphology for needed cells that are nodata
    grid = pd.read_csv(data.GRID_CSV)
    grid, imputed = impute_missing_morphology(grid, set(cell_of.values()))
    if imputed:
        grid.to_csv(data.GRID_CSV, index=False)
        with open(IMPUTED_JSON, 'w', encoding='utf-8') as fh:
            json.dump({'cells': imputed, 'k': KNN_IMPUTE,
                       'method': 'inverse-distance weighted mean of the k nearest complete cells',
                       'reason': 'holds a station in the shared split but is nodata in '
                                 'master_raster_v2.tif (their 10 m raster covers it)'}, fh, indent=2)
        print('  imputed morphology for cell(s) %s -> data/imputed_cells.json' % imputed)

    # ---- partitions
    test_cells = sorted({cell_of[s] for s in TEST_STATIONS if s in cell_of})
    cal_cells = sorted({cell_of[s] for s in CALIBRATION_STATIONS if s in cell_of})
    held = set(test_cells) | set(cal_cells)

    complete_ids = set(grid.loc[grid[MORPH_COLS].notna().all(axis=1), 'id'].astype(int))
    all_cells = set(cell_of.values())
    train_cells = sorted((all_cells - held) & complete_ids)
    dropped = sorted((all_cells - held) - complete_ids)

    print('\npartitions (cells)')
    print('  test        %2d cells from %2d stations' % (len(test_cells), len(TEST_STATIONS)))
    print('  calibration %2d cells from %2d stations' % (len(cal_cells), len(CALIBRATION_STATIONS)))
    print('  train       %2d cells' % len(train_cells))
    if dropped:
        print('  excluded from train (morphology still missing): %s' % dropped)

    # ---- folds inside train, matching their cv_folds: 3
    k_outer, k_inner = 3, 3
    outer_map = spatial_cv.get_outer_folds(k=k_outer, random_state=42,
                                           exclude_grid_ids=sorted(held))
    outer_map = {int(c): int(f) for c, f in outer_map.items() if int(c) in set(train_cells)}
    outer_by_fold = spatial_cv._folds_to_cells(outer_map)

    inner = {}
    for fold, cells in outer_by_fold.items():
        im = spatial_cv.get_inner_folds(k=k_inner, random_state=42,
                                        exclude_grid_ids=sorted(held) + list(cells))
        inner[str(fold)] = {int(c): int(v) for c, v in im.items() if int(c) in set(train_cells)}

    doc = {
        'mapping_csv': None,
        'partitions': {'train': train_cells, 'calibration': cal_cells, 'test': test_cells},
        'info': {
            'source': 'CNN-LSTM run data/20261016_070432 - same 83 stations, their split',
            'test_stations': TEST_STATIONS,
            'calibration_stations': CALIBRATION_STATIONS,
            'imputed_cells': imputed,
            'k_outer': k_outer, 'k_inner': k_inner,
            'note': 'station 9000015 is in TEST and its morphology is imputed from neighbours',
        },
        'folds': {
            'k_outer': k_outer, 'k_inner': k_inner,
            'outer': {str(f): cells for f, cells in outer_by_fold.items()},
            'inner': inner,
            'outer_training_cells': train_cells,
        },
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=2, sort_keys=True, default=str)

    print('\nwrote %s' % OUT_JSON)
    print('  outer folds: %s' % {f: len(c) for f, c in outer_by_fold.items()})
    print('  total cells: %d (train %d + cal %d + test %d)'
          % (len(train_cells) + len(cal_cells) + len(test_cells),
             len(train_cells), len(cal_cells), len(test_cells)))
    print('\n  their split: 54 / 13 / 16 = 83 cells')


if __name__ == '__main__':
    main()
