"""split.cv_assignment: `balanced` (sklearn GroupKFold, balanced by rows) or `by_station` (the same station groups for any dataset)."""
import numpy as np
import pandas as pd
import pytest

import run_pipeline
from config_parser import apply_overrides, validate
from preprocessing.spatial_partitioner import (CV_ASSIGNMENTS, fold_settings, nested_spatial_folds, partition_locations,
                                               spatial_cv_folds)
from tests.test_tune_progress import _tiny_processed
from training import tuning


def _meta(counts: dict) -> pd.DataFrame:
    return pd.DataFrame({"location_id": np.repeat(list(counts), list(counts.values()))})


def _groups(meta, folds) -> frozenset:
    """The station groups of a set of folds (the fold numbering is arbitrary)."""
    return frozenset(frozenset(meta["location_id"].iloc[val].unique()) for _, val in folds)


EVEN = {s: 100 for s in range(1, 10)}
SKEWED = {1: 900, 2: 50, 3: 40, 4: 30, 5: 20, 6: 10, 7: 60, 8: 70, 9: 80}              # the same nine stations, other row counts


def test_the_default_is_the_old_balanced_behaviour(base_cfg):
    assert base_cfg["data"]["split"]["cv_assignment"] == "balanced" and CV_ASSIGNMENTS == ("balanced", "by_station")
    assert fold_settings(base_cfg) == {"assignment": "balanced", "seed": 42}
    meta = _meta(SKEWED)
    from sklearn.model_selection import GroupKFold
    old = list(GroupKFold(3).split(np.zeros(len(meta)), groups=meta["location_id"]))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(spatial_cv_folds(meta, 3), old))


def test_balanced_folds_move_when_the_row_counts_move(base_cfg):
    """The problem by_station solves: the same nine stations, other row counts, other groups."""
    assert _groups(_meta(EVEN), spatial_cv_folds(_meta(EVEN), 3)) != _groups(_meta(SKEWED), spatial_cv_folds(_meta(SKEWED), 3))


def test_by_station_gives_the_same_groups_whatever_the_row_counts():
    a, b = _meta(EVEN), _meta(SKEWED)
    fa, fb = spatial_cv_folds(a, 3, "by_station", 42), spatial_cv_folds(b, 3, "by_station", 42)
    assert _groups(a, fa) == _groups(b, fb)
    for (_, va), (_, vb) in zip(fa, fb):                                                   # and the same numbering
        assert set(a["location_id"].iloc[va]) == set(b["location_id"].iloc[vb])


def test_by_station_folds_hold_out_whole_stations_exactly_once_and_stay_even():
    meta = _meta(SKEWED)
    folds = spatial_cv_folds(meta, 3, "by_station", 42)
    val_stations = [set(meta["location_id"].iloc[v]) for _, v in folds]
    assert sorted(s for v in val_stations for s in v) == sorted(SKEWED)                    # each station validated exactly once
    for fit, val in folds:
        assert not set(meta["location_id"].iloc[fit]) & set(meta["location_id"].iloc[val])      # never in its own training set
        assert len(fit) + len(val) == len(meta) and len(set(fit) & set(val)) == 0
    assert max(map(len, val_stations)) - min(map(len, val_stations)) <= 1                  # 3 / 3 / 3 stations


@pytest.mark.parametrize("n_stations", [10, 11, 17])
def test_folds_stay_within_one_station_of_each_other_when_the_count_does_not_divide(n_stations):
    meta = _meta({s: 20 + s for s in range(1, n_stations + 1)})
    sizes = [len(set(meta["location_id"].iloc[v])) for _, v in spatial_cv_folds(meta, 3, "by_station", 42)]
    assert sum(sizes) == n_stations and max(sizes) - min(sizes) <= 1 and min(sizes) >= 1


def test_by_station_is_seeded_and_the_seed_matters():
    meta = _meta(EVEN)
    same = [spatial_cv_folds(meta, 3, "by_station", 7) for _ in range(2)]
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(*same))
    others = {_groups(meta, spatial_cv_folds(meta, 3, "by_station", s)) for s in range(12)}
    assert len(others) > 3                                                                 # not one fixed arrangement


def test_by_station_needs_at_least_one_station_per_fold_and_a_known_name():
    with pytest.raises(ValueError, match="cannot fill 3"):
        spatial_cv_folds(_meta({1: 5, 2: 5}), 3, "by_station")
    with pytest.raises(ValueError, match="unknown cv_assignment"):
        spatial_cv_folds(_meta(EVEN), 3, "shuffled")


def test_nested_folds_keep_the_outer_test_stations_out_of_the_inner_ones(cfg):
    meta = _meta({s: 50 + s for s in range(1, 19)})
    for outer_fit, outer_test, inner in nested_spatial_folds(meta, 3, 3, "by_station", 42):
        test_ids = set(meta["location_id"].iloc[outer_test])
        assert not test_ids & set(meta["location_id"].iloc[outer_fit])
        for f, v in inner:
            assert set(f) <= set(outer_fit) and set(v) <= set(outer_fit)
            assert not set(meta["location_id"].iloc[f]) & set(meta["location_id"].iloc[v])
    other = _meta({s: 400 - 7 * s for s in range(1, 19)})                                    # other row counts, same stations
    a = [(_groups(meta, [(0, o)]), ) for _, o, _ in nested_spatial_folds(meta, 3, 3, "by_station", 42)]
    b = [(_groups(other, [(0, o)]), ) for _, o, _ in nested_spatial_folds(other, 3, 3, "by_station", 42)]
    assert a == b


def test_the_split_of_stations_is_untouched_by_the_setting(cfg):
    ids = list(range(1, 85))
    before = partition_locations(ids, cfg)
    cfg["data"]["split"]["cv_assignment"] = "by_station"
    assert partition_locations(ids, cfg) == before


def test_an_unknown_assignment_is_refused_when_the_config_is_validated(cfg):
    validate(apply_overrides(cfg, ["data.split.cv_assignment=by_station"]))
    with pytest.raises(ValueError, match="cv_assignment"):
        validate(apply_overrides(cfg, ["data.split.cv_assignment=random"]))


# ------------------------------------------------------------------------------ the stages really use it
class _Stop(Exception):
    pass


def _prepared(small_cfg, tmp_path, assignment):
    processed = tmp_path / "processed"
    processed.mkdir()
    _tiny_processed(small_cfg, processed)
    (processed / "preprocess_info.json").write_text("{}")
    small_cfg["data"]["paths"].update(processed_dir=str(processed), checkpoints_dir=str(tmp_path / "ckpt"),
                                      snapshots_dir=str(tmp_path / "snap"), tuning_dir=str(tmp_path / "tune"))
    small_cfg["data"]["split"].update(cv_assignment=assignment, seed=11, cv_folds=2)
    small_cfg["train"]["device"] = "cpu"
    return small_cfg


def test_train_passes_the_assignment_and_the_seed_to_the_folds(small_cfg, tmp_path, monkeypatch):
    cfg = _prepared(small_cfg, tmp_path, "by_station")
    seen = {}

    def spy(meta, n, assignment="balanced", seed=42):
        seen.update(n=n, assignment=assignment, seed=seed)
        raise _Stop

    monkeypatch.setattr("preprocessing.spatial_partitioner.spatial_cv_folds", spy)
    with pytest.raises(_Stop):
        run_pipeline.train(cfg)
    assert seen == {"n": 2, "assignment": "by_station", "seed": 11}


def test_tuning_passes_the_assignment_to_the_outer_inner_and_final_folds(small_cfg, tmp_path, monkeypatch):
    cfg = _prepared(small_cfg, tmp_path, "by_station")
    cfg["tune"].update(outer_folds=2, inner_folds=2, n_calls=2, n_initial_points=2, row_stride=1, max_epochs=1, patience=1)
    calls = []
    real_nested, real_cv = tuning.nested_spatial_folds, tuning.spatial_cv_folds
    monkeypatch.setattr(tuning, "nested_spatial_folds", lambda *a, **k: calls.append(("nested", k)) or real_nested(*a, **k))
    monkeypatch.setattr(tuning, "spatial_cv_folds", lambda *a, **k: calls.append(("final", k)) or real_cv(*a, **k))
    tuning.nested_tuning(cfg)
    assert [c[0] for c in calls] == ["nested", "final"]
    assert all(k == {"assignment": "by_station", "seed": 11} for _, k in calls)
