import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from preprocessing.multimodal_aligner import stack_patches, station_index
from preprocessing.spatial_patch_extractor import extract_patch, extract_patches
from preprocessing.temporal_window_builder import build_windows


@pytest.fixture
def raster(tmp_path):
    """10x10 pixels, 10 m, 3 bands; band b at (row r, col c) = 1000*b + 10*r + c."""
    path = tmp_path / "r.tif"
    data = np.stack([[[1000 * b + 10 * r + c for c in range(10)] for r in range(10)] for b in range(3)]).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=3, dtype="float32",
                       crs="EPSG:32651", transform=from_origin(0, 100, 10, 10)) as dst:
        dst.write(data)
    return str(path), data


def test_patch_is_the_right_raster_window(raster):
    path, data = raster
    with rasterio.open(path) as src:
        patch = extract_patch(src, x=45, y=55, extent_m=30, resolution_m=10)   # pixel row 4, col 4
    assert patch.shape == (3, 3, 3)
    assert np.array_equal(patch, data[:, 3:6, 3:6])


def test_patch_is_zero_padded_at_the_raster_edge(raster):
    path, data = raster
    with rasterio.open(path) as src:
        patch = extract_patch(src, x=5, y=95, extent_m=30, resolution_m=10)    # pixel row 0, col 0
    assert (patch[:, 0, :] == 0).all() and (patch[:, :, 0] == 0).all()
    assert np.array_equal(patch[:, 1:, 1:], data[:, 0:2, 0:2])


def test_patch_and_window_match_same_station_and_time(raster, small_cfg):
    """Row i: its patch is the raster window of ITS station, its window is that station's readings
    at the lag hours before ITS timestamp."""
    path, data = raster
    small_cfg["data"]["patch"]["extent_m"] = 30
    stations = pd.DataFrame({"location_id": [1, 2], "x": [45.0, 75.0], "y": [55.0, 25.0]})
    patches = extract_patches(path, stations, small_cfg)
    arr, index = stack_patches(patches)

    hours = pd.date_range("2025-01-01", periods=6, freq="h")
    frames = [pd.DataFrame({"location_id": sid, "local_timestamp": hours,
                            "pm25_value": base + np.arange(6.0), "x_coordinate": 0.0, "y_coordinate": 0.0})
              for sid, base in [(1, 0.0), (2, 100.0)]]
    X, y, meta = build_windows(pd.concat(frames, ignore_index=True), small_cfg)

    sidx = station_index(meta, index)
    i = int(np.flatnonzero((meta.location_id == 2) & (meta.timestamp == hours[4]))[0])
    assert np.array_equal(arr[sidx[i]], data[:, 6:9, 6:9])      # station 2 sits at pixel row 7, col 7
    assert not np.array_equal(arr[sidx[i]], data[:, 3:6, 3:6])  # ...and is not station 1's patch
    assert X[i, :, 0].tolist() == [102.0, 103.0]       # t-2, t-1 of station 2 at hour 4
    assert y[i, 0] == 104.0


def test_station_index_raises_for_missing_patch():
    meta = pd.DataFrame({"location_id": [1, 99], "timestamp": pd.to_datetime(["2025-01-01"] * 2)})
    with pytest.raises(KeyError):
        station_index(meta, {1: 0})


def test_windows_drop_gaps_and_never_cross_stations(small_cfg):
    hours = pd.date_range("2025-01-01", periods=8, freq="h")
    a = np.arange(8.0)
    a[4] = np.nan                                   # un-interpolated outage at hour 4
    frames = [pd.DataFrame({"location_id": sid, "local_timestamp": hours, "pm25_value": v,
                            "x_coordinate": 0.0, "y_coordinate": 0.0})
              for sid, v in [(1, a), (2, 100 + np.arange(8.0))]]
    X, y, meta = build_windows(pd.concat(frames, ignore_index=True), small_cfg)

    a_hours = set(meta.loc[meta.location_id == 1, "timestamp"])
    # t needs t, t-1, t-2: hours 4, 5, 6 touch the gap; hours 0, 1 have no history
    assert a_hours == {hours[2], hours[3], hours[7]}
    b = meta.location_id == 2
    assert b.sum() == 6 and (X[b.to_numpy()][:, :, 0] >= 100).all()      # station 2 never sees station 1
