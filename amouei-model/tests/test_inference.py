import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from config_parser import apply_overrides, validate
from inference.cell_inference import (cell_temporal_features, gather_patches, padded_raster, predict_surface,
                                      standin_series)
from preprocessing.scalers import Scalers
from preprocessing.spatial_patch_extractor import add_cell_columns, extract_patch
from uncertainty.geocp import calibration_table


@pytest.fixture
def raster(tmp_path):
    """10x10 pixels, 10 m, 3 bands; band b at (row r, col c) = 1000*b + 10*r + c."""
    path = tmp_path / "r.tif"
    data = np.stack([[[1000 * b + 10 * r + c for c in range(10)] for r in range(10)] for b in range(3)]).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=3, dtype="float32",
                       crs="EPSG:32651", transform=from_origin(0, 100, 10, 10)) as dst:
        dst.write(data)
    return str(path)


def _centre(r, c):
    return c * 10 + 5.0, 95.0 - r * 10


@pytest.mark.parametrize("px", [1, 3, 4])
def test_vectorised_patches_equal_extract_patch_including_edges(raster, px):
    padded, _, _, _ = padded_raster(raster, px)
    pixels = [(0, 0), (0, 9), (9, 0), (9, 9), (4, 5), (1, 8)]
    rows, cols = np.array([p[0] for p in pixels]), np.array([p[1] for p in pixels])
    got = gather_patches(padded, rows, cols, px)
    with rasterio.open(raster) as src:
        for i, (r, c) in enumerate(pixels):
            x, y = _centre(r, c)
            assert np.array_equal(got[i], extract_patch(src, x, y, extent_m=px * 10, resolution_m=10)), (r, c)


def _hourly():
    h = np.arange(10, dtype=float)
    return np.column_stack([10 + h, 30 + h])                       # station 1 and 2 readings per hour


def test_standin_is_the_weighted_mean_of_the_nearest_stations(cfg):
    cfg["data"]["spatial_lag"].update(k=5, weighting="inverse_distance")
    ref_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    out = standin_series(_hourly(), [1, 2], [1, 2], ref_xy, np.array([[25.0, 0.0]]), np.array([0, 1]), cfg)
    assert out[:, 0] == pytest.approx([0.75 * 10 + 0.25 * 30, 0.75 * 11 + 0.25 * 31])
    on_station = standin_series(_hourly(), [1, 2], [1, 2], ref_xy, np.array([[0.0, 0.0]]), np.array([0]), cfg)
    assert on_station[0, 0] == pytest.approx((10 * 1.0 + 30 * 0.01) / 1.01)         # the cell's own sensor dominates


def test_missing_reading_renormalises_and_no_reading_gives_nan(cfg):
    ref_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    v = _hourly()
    v[0, 1] = np.nan
    assert standin_series(v, [1, 2], [1, 2], ref_xy, np.array([[25.0, 0.0]]), np.array([0]), cfg)[0, 0] == pytest.approx(10.0)
    v[0, 0] = np.nan
    assert np.isnan(standin_series(v, [1, 2], [1, 2], ref_xy, np.array([[25.0, 0.0]]), np.array([0]), cfg)[0, 0])


def test_feature_order_matches_training(cfg):
    """[pm25, spatial_lag, sin_hour, cos_hour] per step, oldest lag first; pm25 and spatial_lag both carry the stand-in."""
    cfg["data"]["spatial_lag"]["k"] = 2
    ts = pd.Timestamp("2025-01-01 06:00")
    X = cell_temporal_features(_hourly(), [1, 2], [1, 2], np.array([[0.0, 0.0], [100.0, 0.0]]),
                               np.array([[25.0, 0.0]]), t_idx=6, timestamp=ts, cfg=cfg)
    assert X.shape == (1, 3, 4)
    assert X[0, :, 0].tolist() == pytest.approx([18.0, 19.0, 20.0])        # lag hours 3, 2, 1 -> 15 + hour index
    assert np.allclose(X[0, :, 0], X[0, :, 1])
    for j, lag in enumerate(cfg["data"]["temporal"]["lags"]):
        hour = (ts - pd.Timedelta(hours=lag)).hour
        assert X[0, j, 2] == pytest.approx(np.sin(2 * np.pi * hour / 24))
        assert X[0, j, 3] == pytest.approx(np.cos(2 * np.pi * hour / 24))


class _LastLag(torch.nn.Module):
    """Ignores the patch and returns the (scaled) PM2.5 of the most recent lag step."""
    def forward(self, spatial, temporal):
        return temporal[:, -1, :1]


def _scalers():
    return Scalers().fit(np.ones((2, 3, 1, 1)), np.array([[10.0], [30.0]]), [0, 1], [])        # mean 20, std 10


def test_surface_prediction_for_every_cell(cfg, raster):
    cfg["data"]["spatial_lag"]["k"] = 2
    cfg["derived"]["patch_px"] = 1
    cfg["inference"].update(stride_px=1, chunk_cells=7, window=None)          # 7 does not divide 100: chunking is exercised
    padded, transform, crs, shape = padded_raster(raster, 1)
    hourly = {"values": _hourly(), "location_ids": np.array([1, 2]), "start": pd.Timestamp("2025-01-01 00:00")}
    ref_xy = np.array([_centre(0, 0), _centre(9, 9)])
    pred, out_transform = predict_surface(_LastLag(), _scalers(), padded, shape, transform, hourly, [1, 2], ref_xy,
                                          pd.Timestamp("2025-01-01 06:00"), cfg)
    assert pred.shape == (10, 10) and out_transform == transform
    for r, c in [(0, 0), (9, 9), (4, 5), (0, 9)]:
        d = np.hypot(*(ref_xy - np.array(_centre(r, c))).T)                    # distance to each station
        w = 1.0 / np.maximum(d, 1.0)
        w = w / w.sum()
        expected = w[0] * 15.0 + w[1] * 35.0                                   # readings at hour 5 (t-1): 10+5 and 30+5
        assert pred[r, c] == pytest.approx(expected, rel=1e-4)


def test_no_history_means_no_prediction(cfg, raster):
    cfg["derived"]["patch_px"] = 1
    padded, transform, _, shape = padded_raster(raster, 1)
    values = _hourly()
    values[:, :] = np.nan
    hourly = {"values": values, "location_ids": np.array([1, 2]), "start": pd.Timestamp("2025-01-01 00:00")}
    pred, _ = predict_surface(_LastLag(), _scalers(), padded, shape, transform, hourly, [1, 2],
                              np.array([_centre(0, 0), _centre(9, 9)]), pd.Timestamp("2025-01-01 06:00"), cfg)
    assert np.isnan(pred).all()


def test_stride_and_window_shrink_the_output_grid(cfg, raster):
    cfg["data"]["spatial_lag"]["k"] = 2
    cfg["derived"]["patch_px"] = 1
    cfg["inference"].update(stride_px=3, chunk_cells=50, window=[0, 9, 0, 6])
    padded, transform, _, shape = padded_raster(raster, 1)
    hourly = {"values": _hourly(), "location_ids": np.array([1, 2]), "start": pd.Timestamp("2025-01-01 00:00")}
    pred, out_transform = predict_surface(_LastLag(), _scalers(), padded, shape, transform, hourly, [1, 2],
                                          np.array([_centre(0, 0), _centre(9, 9)]), pd.Timestamp("2025-01-01 06:00"), cfg)
    assert pred.shape == (3, 2) and out_transform.a == 30.0                    # 30 m cells; 3 rows x 2 cols


def test_even_stride_is_refused(cfg):
    cfg = apply_overrides(cfg, ["inference.stride_px=2"])
    with pytest.raises(ValueError):
        validate(cfg)


def test_cell_centres_are_attached_to_stations(raster):
    stations = pd.DataFrame({"location_id": [7], "x": [47.0], "y": [52.0]})
    out = add_cell_columns(stations, raster)
    assert (out["cell_row"][0], out["cell_col"][0]) == (4, 4)
    assert (out["cell_x"][0], out["cell_y"][0]) == (45.0, 55.0)


def test_calibration_table_uses_cell_centres_and_a_cell_id():
    pred = pd.DataFrame({"location_id": [7], "timestamp": pd.to_datetime(["2025-01-01"]), "y_true": [10.0], "y_pred": [7.0]})
    stations = pd.DataFrame({"location_id": [7], "x": [45.0], "y": [55.0], "cell_row": [4], "cell_col": [4]})
    t = calibration_table(pred, stations)
    assert list(t.columns) == ["location_id", "cell_id", "x", "y", "timestamp", "y_true", "y_pred", "score"]
    assert t["cell_id"][0] == "4_4" and t["x"][0] == 45.0 and t["score"][0] == 3.0
