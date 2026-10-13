import numpy as np
import pandas as pd

from preprocessing.spatial_partitioner import partition_indices, partition_locations, spatial_cv_folds


def _ids(n=20):
    return list(range(100, 100 + n))


def test_partitions_are_disjoint_by_location(cfg):
    part = partition_locations(_ids(), cfg)
    sets = [set(v) for v in part.values()]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert set().union(*sets) == set(_ids())
    assert all(len(v) > 0 for v in part.values())


def test_partition_is_seeded(cfg):
    assert partition_locations(_ids(), cfg) == partition_locations(_ids(), cfg)


def test_rows_follow_their_location(cfg):
    part = partition_locations(_ids(), cfg)
    meta = pd.DataFrame({"location_id": np.repeat(_ids(), 5)})
    rows = partition_indices(meta, part)
    for name, idx in rows.items():
        assert set(meta.iloc[idx]["location_id"]) == set(part[name])
    assert sum(len(v) for v in rows.values()) == len(meta)


def test_cv_folds_hold_out_whole_stations(cfg):
    meta = pd.DataFrame({"location_id": np.repeat(_ids(9), 4)})
    folds = spatial_cv_folds(meta, 3)
    assert len(folds) == 3
    for fit, val in folds:
        assert not set(meta.iloc[fit]["location_id"]) & set(meta.iloc[val]["location_id"])


def test_explicit_station_lists_override_fractions(cfg):
    cfg["data"]["split"]["test_stations"] = [100, 101]
    cfg["data"]["split"]["calibration_stations"] = [102]
    part = partition_locations(_ids(), cfg)
    assert part["test"] == [100, 101] and part["calibration"] == [102]
    assert not {100, 101, 102} & set(part["train"])
