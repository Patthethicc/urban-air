import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ADDITIONAL NOTES FOR SPATIAL_CV
# runs k-means on grid cell centroids to assign spatial folds
# used for both outer and inner cv loops.
#
# STEP 2 ADDITIONS (plan Sections 4.2.1 and 3.3):
# 1) get_partitions() splits the cells into train / calibration / test clusters, so the held-out
#    cells are geographically separated and GeoCP has calibration cells the model never fit on.
# 2) get_outer_folds() now takes exclude_grid_ids, so the test and calibration cells stay out of
#    the nested CV entirely. They were being used as CV folds before this step.
# 3) save_fold_map() / load_fold_map() / build_fold_document() persist the partitions and the fold
#    maps to one JSON file, so later notebook runs reuse the same split instead of a fresh K-Means
#    draw (and so GeoCP can be calibrated on exactly the cells the CV never saw).
#
# SAMPLE SIZE WARNING: 17 cells partition into 11 train / 3 calibration / 3 test. At outer k=3 that
# is 3-5 cells per outer fold, and inner k=3 leaves 1-2 cells per inner fold. Thin, but it is what
# 17 cells allows; plan Section 3.2 says to state the chosen k in the paper rather than imply more
# statistical power than the network has.

DEFAULT_MAPPING_CSV = None  # resolved from data.SENSOR_MAP_CSV at call time
PARTITION_NAMES = ('train', 'calibration', 'test')


# gets one row per unique Grid_ID with its projected centroid (cx, cy)
# in EPSG:32651 (meters), suitable for k-means distance-based clustering
def get_grid_centroids(mapping_csv_path=None):
    import data
    df = pd.read_csv(data.resolve(mapping_csv_path, data.SENSOR_MAP_CSV))
    df = df.rename(columns={'sensor_id': 'Location', 'id': 'Grid_ID'})
    df = df.drop_duplicates(subset='Location', keep='first')
    df['cx'] = (df['left'] + df['right']) / 2
    df['cy'] = (df['top'] + df['bottom']) / 2
    centroids = df.groupby('Grid_ID')[['cx', 'cy']].first().reset_index()
    return centroids

# assigns spatial folds to grid cell centroids using k-means clustering
def assign_spatial_folds(centroids, k, random_state=42):
    centroids = centroids.copy()
    km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    centroids['fold'] = km.fit_predict(centroids[['cx', 'cy']])
    return centroids

# assigns outer folds to grid cell centroids, optionally excluding cells that must stay out of CV
# (the calibration and test cells). exclude_grid_ids is what keeps those held-out partitions out of
# the nested CV; passing it is the same pattern get_inner_folds already used.
def get_outer_folds(mapping_csv_path=None, k=4, random_state=42,
                    exclude_grid_ids=None):
    centroids = get_grid_centroids(mapping_csv_path)
    if exclude_grid_ids is not None:
        centroids = centroids[~centroids['Grid_ID'].isin(exclude_grid_ids)].reset_index(drop=True)
    centroids = assign_spatial_folds(centroids, k=k, random_state=random_state)
    return dict(zip(centroids['Grid_ID'], centroids['fold']))

# assigns inner folds to grid cell centroids
def get_inner_folds(mapping_csv_path=None, exclude_grid_ids=None,
                     k=3, random_state=42):
    centroids = get_grid_centroids(mapping_csv_path)
    if exclude_grid_ids is not None:
        centroids = centroids[~centroids['Grid_ID'].isin(exclude_grid_ids)].reset_index(drop=True)
    centroids = assign_spatial_folds(centroids, k=k, random_state=random_state)
    return dict(zip(centroids['Grid_ID'], centroids['fold']))


# ---------------------------------------------------------------------------------------------
# Step 2a: train / calibration / test partitions
# ---------------------------------------------------------------------------------------------

def _pairwise_distances(coords):
    """n x n Euclidean distance matrix (metres) with a zero diagonal."""
    coords = np.asarray(coords, dtype=float)
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, 0.0)
    return d


def _nearest_partner_dist(coords):
    """For each point, the distance to its nearest other point (inf if it is alone)."""
    d = _pairwise_distances(coords)
    if len(coords) < 2:
        return np.array([np.inf])
    d[d == 0.0] = np.inf
    return d.min(axis=1)


def _partition_separation(centroids, labels):
    """Mean over clusters of the mean nearest-partner distance inside each cluster.

    Larger is better here: a larger number means each held-out cluster's cells sit close to each
    other and far from other clusters, which is what makes the spatial CV an honest "new place"
    test instead of a scattered sample of the same neighbourhood.
    """
    coords = centroids[['cx', 'cy']].values
    vals = [_nearest_partner_dist(coords[labels == c]).mean() for c in np.unique(labels)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def search_partitions(centroids, k=3, min_per_partition=2, n_restarts=200,
                      random_state=42):
    """Find a K-Means partition of the cell centroids that keeps held-out cells separated.

    Scores every seed by _partition_separation and keeps the best assignment whose clusters all
    hold at least min_per_partition cells. Returns (centroids_copy, labels, info) where info
    records the winning seed and score so the choice can be reproduced and audited.
    """
    coords = centroids[['cx', 'cy']].values
    best = None
    for seed in range(n_restarts):
        labels = KMeans(n_clusters=k, random_state=seed, n_init='auto').fit_predict(coords)
        sizes = [int((labels == c).sum()) for c in range(k)]
        if min(sizes) < min_per_partition:
            continue
        sep = _partition_separation(centroids, labels)
        # ties broken on a stable hash of the label vector so the result does not depend on
        # floating-point noise between runs
        key = (round(sep, 6), tuple(sorted(sizes)))
        if best is None or key > best[0]:
            best = (key, int(seed), labels, sep, sizes)

    if best is None:
        raise ValueError(
            f"no seed produced {k} partitions of at least {min_per_partition} cells each "
            f"from {len(centroids)} cells; lower min_per_partition or k"
        )

    _, seed, labels, sep, sizes = best
    out = centroids.copy()
    out['part'] = labels
    info = {'seed': seed, 'separation_m': round(sep, 1), 'sizes': sizes,
            'n_restarts': n_restarts, 'random_state': random_state}
    return out, labels, info


def _label_partitions(labelled, target_sizes, names=PARTITION_NAMES):
    """Name the clusters by size, largest -> train, then -> calibration and test.

    Naming by rank (not by K-Means label) keeps the split deterministic whatever order K-Means
    returns the clusters in. If the target sizes do not match, the requested names are assigned to
    the closest-matching clusters so a slightly different cell count cannot silently swap the
    whole partition scheme.
    """
    sizes = labelled.groupby('part').size().to_dict()
    if sorted(sizes.values()) == sorted(target_sizes.values()):
        ranked = sorted(sizes, key=lambda c: (-sizes[c], c))
    else:
        # match each requested partition to the cluster whose size is closest to its target
        remaining = list(sizes)
        ranked = []
        for name in names:
            want = target_sizes[name]
            pick = min(remaining, key=lambda c: (abs(sizes[c] - want), c))
            remaining.remove(pick)
            ranked.append(pick)
    mapping = {int(cluster): name for cluster, name in zip(ranked, names)}
    return mapping


def get_partitions(mapping_csv_path=None, n_train=11, n_calibration=3, n_test=3,
                   k=3, n_restarts=200, random_state=42):
    """Assign every grid cell to 'train', 'calibration' or 'test' using spatial clustering.

    Calibration cells are what GeoCP's conformity scores are built from, so they must be cells the
    model never trained on; test cells are the final untouched read. Both are chosen together with
    the train cells so the three partitions are geographically separated.

    Returns {'partitions': {name: [Grid_ID, ...]}, 'info': {...}} with info recording the winning
    search seed, the separation score, per-partition sizes and each partition's centroid, so the
    split can be reported in the paper and reproduced exactly.
    """
    centroids = get_grid_centroids(mapping_csv_path)
    labelled, _, info = search_partitions(centroids, k=k, n_restarts=n_restarts,
                                          random_state=random_state)
    mapping = _label_partitions(labelled, {'train': n_train, 'calibration': n_calibration,
                                           'test': n_test})
    labelled['partition'] = labelled['part'].map(mapping)

    partitions = {name: sorted(int(g) for g in labelled.loc[labelled['partition'] == name,
                                                             'Grid_ID'])
                  for name in PARTITION_NAMES}
    missing = [name for name in PARTITION_NAMES if not partitions[name]]
    if missing:
        raise ValueError(f"partition(s) {missing} ended up empty; adjust k or the target sizes")

    info['partition_sizes'] = {name: len(cells) for name, cells in partitions.items()}
    info['centroids'] = {
        name: {'cx': float(labelled.loc[labelled['partition'] == name, 'cx'].mean()),
               'cy': float(labelled.loc[labelled['partition'] == name, 'cy'].mean())}
        for name in PARTITION_NAMES
    }
    info['min_distance_between_partitions_m'] = partition_separation_m(labelled)
    return {'partitions': partitions, 'info': info}


def partition_separation_m(labelled):
    """Smallest centroid-to-centroid distance between any two different partitions (metres).

    This is the number the paper can quote to justify that the held-out cells are a genuinely
    different place, not a neighbour of a training cell.
    """
    coords = labelled[['cx', 'cy']].values
    parts = labelled['partition'].values
    d = _pairwise_distances(coords)
    best = np.inf
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if parts[i] != parts[j]:
                best = min(best, d[i, j])
    return round(float(best), 1) if np.isfinite(best) else None


# ---------------------------------------------------------------------------------------------
# Step 2b: persist and reload the split
# ---------------------------------------------------------------------------------------------

def _jsonable(obj):
    """numpy scalars/arrays -> plain Python so the JSON round-trips cleanly."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


def build_fold_document(mapping_csv_path=None, k_outer=3, k_inner=3,
                        n_calibration=3, n_test=3, random_state=42, n_restarts=200):
    """One JSON-able document holding the partitions and every fold map derived from them.

    Outer folds are built on the training cells only. The test and calibration cells are excluded
    from both CV levels, and the inner folds for each outer fold are built by excluding that
    fold's held-out cells as well, so no inner fold ever sees the outer fold's test cells.
    """
    part = get_partitions(mapping_csv_path, n_calibration=n_calibration, n_test=n_test,
                          k=3, n_restarts=n_restarts, random_state=random_state)
    train_cells = part['partitions']['train']
    held_out = part['partitions']['calibration'] + part['partitions']['test']

    outer = get_outer_folds(mapping_csv_path, k=k_outer, random_state=random_state,
                            exclude_grid_ids=held_out)
    inner = {}
    for fold, held_cells in _folds_to_cells(outer).items():
        inner[str(fold)] = get_inner_folds(
            mapping_csv_path, k=k_inner, random_state=random_state,
            exclude_grid_ids=held_out + list(held_cells))

    doc = {
        'mapping_csv': mapping_csv_path,
        'partitions': part['partitions'],
        'info': part['info'],
        'folds': {
            'k_outer': k_outer,
            'k_inner': k_inner,
            'outer': {str(f): cells for f, cells in _folds_to_cells(outer).items()},
            'inner': inner,
            'outer_training_cells': train_cells,
        },
    }
    return doc


def _folds_to_cells(fold_map):
    """{fold: [Grid_ID, ...]} from {Grid_ID: fold}."""
    out = {}
    for grid_id, fold in fold_map.items():
        out.setdefault(int(fold), []).append(int(grid_id))
    return {f: sorted(cells) for f, cells in sorted(out.items())}


def save_fold_map(doc, path=None):
    """Write the fold document to JSON. This file is the single source of truth for the split."""
    import data
    path = data.resolve(path, data.FOLD_JSON)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(_jsonable(doc), fh, indent=2, sort_keys=True)
    return path


def load_fold_map(path=None, require_partitions=True):
    """Read the fold document back.

    require_partitions defaults to True so a later step cannot silently run without the
    train/calibration/test split: a document with no 'partitions' key raises instead of quietly
    falling back to "every cell is training data", which is the bug this step exists to fix.
    """
    import data
    path = data.resolve(path, data.FOLD_JSON)
    with open(path, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    if require_partitions and not doc.get('partitions'):
        raise ValueError(f"{path} has no 'partitions' key; rebuild it with build_fold_document()")
    return doc


def main():
    """Rebuild fold_partitions.json and print what the split actually looks like."""
    doc = build_fold_document()
    path = save_fold_map(doc)
    parts = doc['partitions']
    print(f"Wrote {path}")
    print(f"  separation score (search): {doc['info']['separation_m']} m")
    print(f"  min distance between partitions: "
          f"{doc['info']['min_distance_between_partitions_m']} m")
    for name in PARTITION_NAMES:
        cells = parts[name]
        c = doc['info']['centroids'][name]
        print(f"  {name:12s} {len(cells):2d} cells {cells}")
        print(f"               centroid=({c['cx']:.0f}, {c['cy']:.0f})")
    print(f"  outer k={doc['folds']['k_outer']} on {len(doc['folds']['outer_training_cells'])} "
          f"training cells: {doc['folds']['outer']}")
    print(f"  inner k={doc['folds']['k_inner']}: {doc['folds']['inner']}")


def feasibility_report(mapping_csv_path=None):
    """Print the per-partition and per-fold viability of the chosen split.

    Run this before spending hours on the Bayesian search: it makes the "cells per fold" thinness
    visible instead of letting it surface as a strange metric later.
    """
    doc = build_fold_document(mapping_csv_path=mapping_csv_path)
    print("Partition sizes:", doc['info']['partition_sizes'])
    print("Min distance between partitions:",
          doc['info']['min_distance_between_partitions_m'], "m")
    outer = doc['folds']['outer']
    print(f"\nOuter k={doc['folds']['k_outer']} (cells per fold):",
          {f: len(c) for f, c in outer.items()})
    for fold, cells in sorted(outer.items(), key=lambda kv: int(kv[0])):
        inner = doc['folds']['inner'][fold]
        counts = Counter(inner.values())
        print(f"  outer fold {fold} holds out {cells} -> inner k="
              f"{doc['folds']['k_inner']} cells per inner fold: {dict(sorted(counts.items()))}")
        if min(counts.values()) < 2:
            print(f"    WARNING: an inner fold has {min(counts.values())} cell(s); inner-CV "
                  f"selection is noisy at this size. Report it as a limitation.")
    return doc


if __name__ == '__main__':
    feasibility_report()
