"""Partition by monitoring LOCATION. A location (with all its rows, lag sequences, patches) lives in
exactly one partition.

    all locations
      |- final spatial test      (untouched until models + GeoCP config are final)
      `- development
           |- proper training    (nested spatial CV, hyperparameter search, final fit)
           `- calibration        (GeoCP nonconformity scores only; never used to fit models)

Runs BEFORE spatial_lag, because the lag reference set is the training partition.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def partition_locations(location_ids, cfg: dict) -> dict[str, list]:
    """Return {'train': [...ids], 'calibration': [...], 'test': [...]}, seeded and mutually exclusive.
    Test is carved out first, then calibration from the remainder. split.test_stations /
    calibration_stations are honoured when given."""
    s = cfg["data"]["split"]
    ids = sorted(int(i) for i in set(location_ids))
    rng = np.random.default_rng(s["seed"])

    def pick(pool: list[int], explicit: list, fraction: float) -> list[int]:
        if explicit:
            chosen = [int(i) for i in explicit]
            missing = set(chosen) - set(pool)
            if missing:
                raise ValueError(f"stations not available for this partition: {sorted(missing)}")
            return sorted(chosen)
        n = max(1, int(round(fraction * len(pool))))
        return sorted(int(i) for i in rng.choice(pool, size=n, replace=False))

    test = pick(ids, s["test_stations"], s["test_fraction"])
    development = [i for i in ids if i not in set(test)]
    calibration = pick(development, s["calibration_stations"], s["calibration_fraction"])
    train = [i for i in development if i not in set(calibration)]

    if len(train) < s["cv_folds"]:
        raise ValueError(f"only {len(train)} training stations for {s['cv_folds']} spatial CV folds")
    return {"train": train, "calibration": calibration, "test": test}


def partition_indices(meta: pd.DataFrame, location_partition: dict) -> dict[str, np.ndarray]:
    """Row positions of each partition, from meta.location_id."""
    return {
        name: np.flatnonzero(meta["location_id"].isin(ids).to_numpy())
        for name, ids in location_partition.items()
    }


def spatial_cv_folds(meta_train: pd.DataFrame, n_splits: int):
    """GroupKFold on location_id inside the training partition -> [(fit_idx, val_idx), ...]
    (positions within meta_train). Whole stations are held out per fold."""
    groups = meta_train["location_id"].to_numpy()
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def nested_spatial_folds(meta_train: pd.DataFrame, outer_k: int, inner_k: int):
    """Nested spatial CV inside the training partition.

    Returns [(outer_fit_idx, outer_test_idx, inner_folds), ...] where inner_folds is
    [(inner_fit_idx, inner_val_idx), ...]. All indices are positions in meta_train; the inner folds
    only ever contain outer_fit rows, so the outer test stations never influence the search."""
    nested = []
    for outer_fit, outer_test in spatial_cv_folds(meta_train, outer_k):
        sub = meta_train.iloc[outer_fit]
        n_stations = sub["location_id"].nunique()
        if n_stations < inner_k:
            raise ValueError(f"outer fold has {n_stations} stations, fewer than inner_folds={inner_k}")
        inner = [(outer_fit[f], outer_fit[v]) for f, v in spatial_cv_folds(sub, inner_k)]
        nested.append((outer_fit, outer_test, inner))
    return nested
