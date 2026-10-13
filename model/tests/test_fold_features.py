import numpy as np
import pandas as pd
import pytest

from preprocessing.spatial_lag import add_spatial_lag, gather_row_lags, lag_matrix, neighbour_table, station_coordinates


def _setup(cfg, n_hours=6):
    """Four stations on a line (ids 1-4 at x = 0, 100, 200, 400) with different constant levels."""
    cfg["data"]["spatial_lag"].update(k=2, weighting="inverse_distance")
    xs = {1: 0.0, 2: 100.0, 3: 200.0, 4: 400.0}
    level = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
    hours = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    rows = [pd.DataFrame({"location_id": i, "local_timestamp": hours, "pm25_value": level[i] + np.arange(n_hours),
                          "x_coordinate": xs[i], "y_coordinate": 0.0}) for i in xs]
    return pd.concat(rows, ignore_index=True)


def test_lag_matrix_agrees_with_add_spatial_lag(cfg):
    df = _setup(cfg)
    part = {"train": [1, 2, 3, 4], "calibration": [], "test": []}
    expected = add_spatial_lag(df, part, cfg)
    table = neighbour_table(station_coordinates(df, cfg), part["train"], cfg)
    wide = df.pivot(index="local_timestamp", columns="location_id", values="pm25_value")
    mat = lag_matrix(wide.to_numpy(), wide.columns, table, [1, 2, 3, 4])
    got = expected.pivot(index="local_timestamp", columns="location_id", values="spatial_lag").to_numpy()
    assert np.allclose(mat, got, equal_nan=True)


def test_fold_lag_ignores_validation_stations(cfg):
    """Station 4 is held out. Changing its readings must not change anyone's fold lag, and
    no station may list it as a neighbour."""
    df = _setup(cfg)
    fit = [1, 2, 3]
    coords = station_coordinates(df, cfg)
    table = neighbour_table(coords, fit, cfg)
    assert all(4 not in nb for nb, _ in table.values())

    wide = df.pivot(index="local_timestamp", columns="location_id", values="pm25_value")
    base = lag_matrix(wide.to_numpy(), wide.columns, table, [1, 2, 3, 4])
    wide4 = wide.copy()
    wide4[4] = 999.0
    changed = lag_matrix(wide4.to_numpy(), wide4.columns, table, [1, 2, 3, 4])
    assert np.allclose(base, changed)
    assert not np.isnan(base[:, 3]).any()                  # the held-out station still gets a lag, from fit stations


def test_full_reference_would_have_leaked(cfg):
    """With all stations as reference, station 3's lag DOES depend on station 4: the leak the fold fix removes."""
    df = _setup(cfg)
    cfg["data"]["spatial_lag"]["k"] = 3                    # station 3 then has stations 1, 2 and 4 as neighbours
    coords = station_coordinates(df, cfg)
    wide = df.pivot(index="local_timestamp", columns="location_id", values="pm25_value")
    leaky = neighbour_table(coords, [1, 2, 3, 4], cfg)
    base = lag_matrix(wide.to_numpy(), wide.columns, leaky, [3])
    wide4 = wide.copy()
    wide4[4] = 999.0
    assert not np.allclose(base, lag_matrix(wide4.to_numpy(), wide4.columns, leaky, [3]))


def test_gather_row_lags_picks_the_lagged_hours(cfg):
    lag_mat = np.arange(12.0).reshape(6, 2)                # [hours, stations]
    out = gather_row_lags(lag_mat, col_pos=np.array([1, 0]), t_idx=np.array([4, 5]), lags=[2, 1])
    assert out.tolist() == [[lag_mat[2, 1], lag_mat[3, 1]], [lag_mat[3, 0], lag_mat[4, 0]]]
