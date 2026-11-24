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


def test_with_geocp_off_there_is_no_calibration_partition_and_the_test_stations_stay_the_same(cfg):
    on = partition_locations(_ids(), cfg)
    cfg["geocp"]["enabled"] = False
    off = partition_locations(_ids(), cfg)
    assert off["calibration"] == []
    assert off["test"] == on["test"]                                         # the same test stations, so the runs compare
    assert off["train"] == sorted(on["train"] + on["calibration"])           # the calibration stations now train


def test_a_train_subset_keeps_the_test_stations_and_nests_for_growing_n(cfg):
    cfg["geocp"]["enabled"] = False
    full = partition_locations(_ids(40), cfg)
    sizes = {}
    for n in (6, 12, 20):
        cfg["data"]["split"]["train_subset"] = {"n_stations": n, "seed": 3}
        part = partition_locations(_ids(40), cfg)
        sizes[n] = part
        assert len(part["train"]) == n and part["test"] == full["test"] and set(part["train"]) <= set(full["train"])
        assert set(part) == {"train", "calibration", "test"}                      # the left-out stations are in no partition
    assert set(sizes[6]["train"]) <= set(sizes[12]["train"]) <= set(sizes[20]["train"])      # nested for one seed
    cfg["data"]["split"]["train_subset"] = {"n_stations": 6, "seed": 4}
    assert partition_locations(_ids(40), cfg)["train"] != sizes[6]["train"]       # another seed, another subset


def test_a_train_subset_as_large_as_the_training_set_changes_nothing(cfg):
    cfg["geocp"]["enabled"] = False
    full = partition_locations(_ids(40), cfg)
    cfg["data"]["split"]["train_subset"] = {"n_stations": 10_000, "seed": 1}
    assert partition_locations(_ids(40), cfg) == full
