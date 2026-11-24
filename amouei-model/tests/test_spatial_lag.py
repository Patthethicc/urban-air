import numpy as np
import pandas as pd
import pytest

from preprocessing.spatial_lag import add_spatial_lag, neighbour_table, station_coordinates


def _frame(cfg, readings: dict):
    """Three stations on a line: A at x=0, B at x=100, C at x=300. readings[id] = list per hour."""
    x = {"A": 0.0, "B": 100.0, "C": 300.0}
    ids = {"A": 1, "B": 2, "C": 3}
    hours = pd.date_range("2025-01-01", periods=2, freq="h")
    rows = []
    for name, vals in readings.items():
        rows.append(pd.DataFrame({"location_id": ids[name], "local_timestamp": hours, "pm25_value": vals,
                                  "x_coordinate": x[name], "y_coordinate": 0.0}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def lag_cfg(cfg):
    cfg["data"]["spatial_lag"].update(k=2, weighting="inverse_distance")
    return cfg


PARTITION = {"train": [2, 3], "calibration": [], "test": [1]}      # A is a test station


def test_station_is_never_its_own_neighbour(lag_cfg):
    df = _frame(lag_cfg, {"A": [0, 0], "B": [0, 0], "C": [0, 0]})
    table = neighbour_table(station_coordinates(df, lag_cfg), PARTITION["train"], lag_cfg)
    for sid, (nb, _) in table.items():
        assert sid not in nb


def test_reference_set_is_train_only(lag_cfg):
    df = _frame(lag_cfg, {"A": [0, 0], "B": [0, 0], "C": [0, 0]})
    table = neighbour_table(station_coordinates(df, lag_cfg), PARTITION["train"], lag_cfg)
    for nb, _ in table.values():
        assert set(nb) <= set(PARTITION["train"])


def test_lag_value_matches_hand_computation(lag_cfg):
    """A's neighbours: B (100 m) and C (300 m) -> weights 0.75 / 0.25 -> 0.75*10 + 0.25*30 = 15.
    B only has C left as a neighbour -> 30."""
    df = _frame(lag_cfg, {"A": [5.0, 5.0], "B": [10.0, 10.0], "C": [30.0, 30.0]})
    out = add_spatial_lag(df, PARTITION, lag_cfg)
    get = lambda sid: out.loc[out.location_id == sid, "spatial_lag"].iloc[0]
    assert get(1) == pytest.approx(15.0)
    assert get(2) == pytest.approx(30.0)


def test_missing_neighbour_renormalises_weights(lag_cfg):
    df = _frame(lag_cfg, {"A": [5.0, 5.0], "B": [10.0, 10.0], "C": [np.nan, 30.0]})
    out = add_spatial_lag(df, PARTITION, lag_cfg)
    a = out[out.location_id == 1].sort_values("local_timestamp")["spatial_lag"].tolist()
    assert a[0] == pytest.approx(10.0)          # hour 0: only B reported
    assert a[1] == pytest.approx(15.0)


def test_test_station_readings_never_reach_anyone(lag_cfg):
    base = _frame(lag_cfg, {"A": [5.0, 5.0], "B": [10.0, 10.0], "C": [30.0, 30.0]})
    changed = _frame(lag_cfg, {"A": [999.0, 999.0], "B": [10.0, 10.0], "C": [30.0, 30.0]})
    lag = lambda d: add_spatial_lag(d, PARTITION, lag_cfg).sort_values(["location_id", "local_timestamp"])["spatial_lag"]
    assert np.allclose(lag(base).to_numpy(), lag(changed).to_numpy())
