import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from config_parser import check_raster_crs
from preprocessing.master_raster_builder import compute_height_std
from reporting.spatial_bias import band_names_for


def test_height_std_is_zero_on_flat_ground_and_positive_at_a_building_edge(cfg):
    cfg["data"]["raster"].update(height_std_neighborhood_m=50, resolution_m=10)           # 11 x 11 window
    flat = np.full((20, 20), 12.0, dtype="float32")
    assert compute_height_std(flat, cfg).max() == pytest.approx(0.0, abs=1e-3)
    step = np.zeros((20, 20), dtype="float32")
    step[:, 10:] = 20.0
    std = compute_height_std(step, cfg)
    assert std[:, 0].max() == pytest.approx(0.0, abs=1e-3) and std[10, 9] > 5.0 and std[10, 10] > 5.0


def test_height_std_matches_a_hand_computed_window(cfg):
    cfg["data"]["raster"].update(height_std_neighborhood_m=10, resolution_m=10)           # 3 x 3 window
    h = np.zeros((5, 5), dtype="float32")
    h[2, 2] = 9.0
    window = np.zeros(9)
    window[4] = 9.0
    assert compute_height_std(h, cfg)[2, 2] == pytest.approx(window.std(), rel=1e-3)


def test_sigma_h_is_a_band_of_the_production_raster(base_cfg):
    assert "height_std" in base_cfg["data"]["raster"]["band_order"]


def _raster(tmp_path, count, crs="EPSG:32651", names=None):
    path = tmp_path / "r.tif"
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=count, dtype="float32", crs=crs,
                       transform=from_origin(0, 40, 10, 10)) as dst:
        dst.write(np.zeros((count, 4, 4), "float32"))
        if names:
            dst.descriptions = tuple(names)
    return str(path)


def test_a_five_band_pilot_raster_keeps_the_leading_names(cfg, tmp_path):
    assert band_names_for(_raster(tmp_path, 5), cfg) == ["height", "ndvi", "building_density", "street_width", "aspect_ratio"]
    assert band_names_for(_raster(tmp_path, 6), cfg)[-1] == "height_std"


def test_raster_crs_must_equal_the_pipeline_crs(cfg, tmp_path):
    cfg["data"]["paths"]["master_raster"] = _raster(tmp_path, 3, crs=cfg["data"]["crs"])
    check_raster_crs(cfg)
    cfg["data"]["paths"]["master_raster"] = _raster(tmp_path, 3, crs="EPSG:4326" if cfg["data"]["crs"] != "EPSG:4326" else "EPSG:32651")
    with pytest.raises(ValueError, match="one CRS"):
        check_raster_crs(cfg)
