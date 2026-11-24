import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from uncertainty.geocp import GeoCP
from uncertainty.geocp_surface import BANDS, build_interval_rasters

XY = pd.DataFrame({"location_id": [1, 2], "x": [15.0, 85.0], "y": [85.0, 15.0]})


def _prediction_raster(tmp_path, nan_at=None):
    pred = np.linspace(2, 30, 100).reshape(10, 10).astype("float32")
    if nan_at:
        pred[nan_at] = np.nan
    path = tmp_path / "pred.tif"
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32",
                       crs="EPSG:32651", transform=from_origin(0, 100, 10, 10), nodata=np.nan) as dst:
        dst.write(pred, 1)
    return str(path), pred


def _model():
    rng = np.random.default_rng(0)
    cal = pd.DataFrame({"location_id": rng.choice([1, 2], 300), "score": np.round(np.abs(rng.normal(0, 3, 300)), 2)})
    return GeoCP(0.1, "gaussian", 40.0).fit(cal, XY)


def test_four_aligned_layers_with_the_right_relations(tmp_path):
    src, pred = _prediction_raster(tmp_path)
    out = tmp_path / "surface.tif"
    stats = build_interval_rasters(src, _model(), str(out), lower_bound=0.0, expected_crs="EPSG:32651")
    with rasterio.open(out) as r, rasterio.open(src) as s:
        assert r.count == 4 and r.descriptions == tuple(BANDS)
        assert (r.crs, r.transform, r.shape) == (s.crs, s.transform, s.shape)         # same extent / resolution / CRS
        prediction, lower, upper, width = [r.read(i) for i in range(1, 5)]
    assert np.allclose(prediction, pred)
    assert (lower >= 0).all() and np.allclose(width, upper - lower)
    assert (upper >= prediction).all() and stats["cells_with_interval"] == 100


def test_cell_value_matches_the_point_computation(tmp_path):
    src, pred = _prediction_raster(tmp_path)
    out = tmp_path / "surface.tif"
    model = _model()
    build_interval_rasters(src, model, str(out), n_grid=5000)
    with rasterio.open(out) as r:
        upper = r.read(3)
    q = model.quantile((15.0, 85.0))                      # centre of the top-left cell (row 1, col 1)
    assert upper[1, 1] == pytest.approx(pred[1, 1] + q, rel=1e-5)


def test_nodata_cells_stay_empty(tmp_path):
    src, _ = _prediction_raster(tmp_path, nan_at=(0, 0))
    out = tmp_path / "surface.tif"
    stats = build_interval_rasters(src, _model(), str(out))
    with rasterio.open(out) as r:
        assert np.isnan(r.read(4)[0, 0])
    assert stats["cells"] == 99


def test_wrong_crs_is_refused(tmp_path):
    src, _ = _prediction_raster(tmp_path)
    with pytest.raises(ValueError):
        build_interval_rasters(src, _model(), str(tmp_path / "x.tif"), expected_crs="EPSG:4326")
