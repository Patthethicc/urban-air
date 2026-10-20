import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from config_parser import load_config, validate
from inference.cell_inference import padded_raster
from training.checkpoint import check_compatibility


def test_utm_51n_is_the_one_coordinate_system(base_cfg):
    assert base_cfg["data"]["crs"] == "EPSG:32651"
    assert "crs_profile" not in base_cfg["data"] and "crs_profiles" not in base_cfg["data"]


def test_the_default_raster_is_the_data_folders_master_raster(base_cfg):
    from pathlib import Path
    from rasterio.crs import CRS
    path = Path(base_cfg["data"]["paths"]["master_raster"])
    assert path.name == "master_raster_v2.tif" and path.parent.name == "Morphology" and path.is_file()
    with rasterio.open(path) as r:
        assert r.crs == CRS.from_epsg(32651) and r.res == (10.0, 10.0) and r.count == 5
        assert "building_height_stddev" in r.descriptions                          # sigma_H is in the raster


def test_outputs_have_no_per_version_sub_folders(base_cfg):
    from pathlib import Path
    for key, name in (("processed_dir", "processed"), ("checkpoints_dir", "checkpoints"),
                      ("snapshots_dir", "snapshots"), ("tuning_dir", "tuning")):
        assert Path(base_cfg["data"]["paths"][key]).name == name, key


def test_the_removed_setting_is_an_unknown_key():
    with pytest.raises(KeyError):
        load_config(["data.crs_profile=utm51n"])


def test_a_geographic_crs_is_refused_because_distances_must_be_metres(cfg):
    cfg["data"]["crs"] = "EPSG:4326"
    with pytest.raises(ValueError, match="metres"):
        validate(cfg)
    cfg["data"]["crs"] = "EPSG:32651"
    validate(cfg)


def test_the_output_grid_inherits_the_input_rasters_crs(tmp_path):
    """Input and output share one CRS: the padded raster used for inference carries the Master GeoTIFF's CRS,
    and the prediction / interval rasters are written with that CRS (see test_geocp_surface for the interval layers)."""
    path = tmp_path / "r.tif"
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=2, dtype="float32", crs="EPSG:32651",
                       transform=from_origin(0, 40, 10, 10)) as dst:
        dst.write(np.zeros((2, 4, 4), "float32"))
    _, transform, crs, _ = padded_raster(str(path), 3)
    assert crs.to_epsg() == 32651 and transform == from_origin(0, 40, 10, 10)


def _info(tmp_path, cfg, crs):
    d = cfg["derived"]
    (tmp_path / "scalers.json").write_text("{}")
    (tmp_path / "preprocess_info.json").write_text(json.dumps({
        **{k: d[k] for k in ("T", "F", "C", "patch_px", "feature_names")}, "temporal": {"lags": [3, 2, 1]}, "crs": crs}))
    return str(tmp_path)


def test_a_model_trained_in_one_crs_refuses_data_in_another(cfg, tmp_path):
    ckpt = {"extra": {"derived": cfg["derived"], "lags": [3, 2, 1], "crs": "EPSG:32651"}}
    check_compatibility(ckpt, _info(tmp_path, cfg, "EPSG:32651"))
    with pytest.raises(ValueError, match="crs"):
        check_compatibility(ckpt, _info(tmp_path, cfg, "EPSG:32652"))
