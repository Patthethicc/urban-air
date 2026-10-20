import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.transform import from_origin, xy

from affine_compat import cell_centers
from preprocessing.clean_loader import load_sensor_locations
from preprocessing.coverage import check_coverage, coverage_report, format_coverage

ORIGIN_X, TOP_Y = 280_000.0, 1_610_000.0           # a 1 km x 1 km raster, 10 m pixels, in UTM 51N
TO_LL = Transformer.from_crs(32651, 4326, always_xy=True)


def _sensor(i, name, x, y):
    lon, lat = TO_LL.transform(x, y)
    return {"name": name, "location_id": i, "latitude": lat, "longitude": lon}


@pytest.fixture
def world(cfg, tmp_path):
    raster = tmp_path / "r.tif"
    with rasterio.open(raster, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32", crs="EPSG:32651",
                       transform=from_origin(ORIGIN_X, TOP_Y, 10, 10)) as dst:
        dst.write(np.ones((1, 100, 100), "float32"))
    sensors = pd.DataFrame([
        _sensor(1, "centre", ORIGIN_X + 500, TOP_Y - 500),
        _sensor(2, "two km south", ORIGIN_X + 500, TOP_Y - 1000 - 2000),
        _sensor(3, "300 m east", ORIGIN_X + 1000 + 300, TOP_Y - 500)])
    sensors.to_csv(tmp_path / "loc.csv", index=False)
    cfg["data"]["paths"].update(master_raster=str(raster), sensor_locations_csv=str(tmp_path / "loc.csv"),
                                sensors_gpkg=None, boundary=None)
    return cfg


def test_sensors_outside_the_raster_are_found_with_side_and_distance(world):
    r = coverage_report(world)
    assert r["sensors"]["total"] == 3 and r["sensors"]["inside"] == 1 and not r["complete"]
    out = {o["name"]: o for o in r["sensors"]["outside"]}
    assert out["two km south"]["sides"] == ["south"] and out["two km south"]["outside_by_m"] == pytest.approx(2000, abs=5)
    assert out["300 m east"]["sides"] == ["east"] and out["300 m east"]["outside_by_m"] == pytest.approx(300, abs=5)


def test_the_report_names_what_is_cropped_away(world):
    text = format_coverage(coverage_report(world))
    assert "OUTSIDE (skipped): two km south" in text and "beyond the south edge" in text and "INCOMPLETE" in text


def test_strict_mode_stops_when_coverage_is_incomplete_and_default_only_reports(world, capsys):
    check_coverage(world)                                       # default: report only
    assert "INCOMPLETE" in capsys.readouterr().out
    world["data"]["coverage"]["strict"] = True
    with pytest.raises(ValueError, match="does not cover"):
        check_coverage(world, verbose=False)


def test_full_coverage_is_reported_as_complete_and_never_blocks(world, tmp_path):
    pd.DataFrame([_sensor(1, "a", ORIGIN_X + 500, TOP_Y - 500), _sensor(2, "b", ORIGIN_X + 20, TOP_Y - 20)]).to_csv(
        tmp_path / "loc.csv", index=False)
    world["data"]["coverage"]["strict"] = True
    r = check_coverage(world, verbose=False)
    assert r["complete"] and r["sensors"]["outside"] == []


def test_a_boundary_larger_than_the_raster_reports_the_covered_share_and_the_missing_side(world, tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    gpd.GeoDataFrame(geometry=[box(ORIGIN_X, TOP_Y - 1500, ORIGIN_X + 1000, TOP_Y)], crs="EPSG:32651").to_file(
        tmp_path / "ncr.geojson", driver="GeoJSON")
    world["data"]["paths"]["boundary"] = str(tmp_path / "ncr.geojson")
    b = coverage_report(world)["boundary"]
    assert b["covered_fraction"] == pytest.approx(1000 / 1500, abs=1e-3)
    assert b["missing_km"] == {"south": pytest.approx(0.5, abs=1e-3)}


def test_the_loader_skips_outside_sensors_exactly_as_the_report_says(world):
    assert load_sensor_locations(world)["location_id"].tolist() == [1]


def test_the_default_config_points_into_the_data_folder_and_the_files_exist(base_cfg):
    from pathlib import Path
    p = base_cfg["data"]["paths"]

    def listed(value):                                  # each sensor path is one path or a list of them
        return [Path(v) for v in ([value] if isinstance(value, str) else value)]

    csvs, dirs = listed(p["sensor_locations_csv"]), listed(p["sensor_csv_dir"])
    assert all(c.is_file() for c in csvs) and all(d.is_dir() for d in dirs)
    assert any(c.parent.name == "OpenAQ" for c in csvs)                              # OpenAQ is always read ...
    assert any(d.name == "OpenAQ" and any(d.glob("openaq_*_*.csv")) for d in dirs)  # ... QC data may be added next to it
    assert "pilot" not in " ".join(str(x) for v in p.values() for x in ([v] if not isinstance(v, list) else v)).lower()
    d = base_cfg["derived"]
    assert d["C"] == 5 and d["patch_px"] == 32 and d["raster_resolution_m"] == 10.0


# ------------------------------------------------------------------------------ vectorised pixel centres
@pytest.mark.parametrize("transform", [from_origin(273566.14, 1631737.86, 10, 10), Affine(10, 1.5, 100, 2, -10, 500)])
def test_vectorised_pixel_centres_equal_rasterios(transform):
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, 3897, 500), rng.integers(0, 2729, 500)
    xs, ys = cell_centers(transform, rows, cols)
    rx, ry = xy(transform, rows, cols, offset="center")
    assert np.allclose(xs, rx) and np.allclose(ys, ry)


# ------------------------------------------------------------------------------ progress / ETA on big grids
def _predict(cfg, tmp_path, size):
    import torch
    from inference.cell_inference import padded_raster, predict_surface
    from preprocessing.scalers import Scalers

    path = tmp_path / f"g{size}.tif"
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=3, dtype="float32", crs="EPSG:32651",
                       transform=from_origin(0, size * 10, 10, 10)) as dst:
        dst.write(np.ones((3, size, size), "float32"))

    class Net(torch.nn.Module):
        def forward(self, spatial, temporal):
            return temporal[:, -1, :1]

    cfg["data"]["spatial_lag"]["k"] = 2
    cfg["derived"]["patch_px"] = 1
    cfg["inference"].update(stride_px=1, chunk_cells=4096, window=None, progress=True)
    scalers = Scalers().fit(np.ones((2, 3, 1, 1)), np.array([[10.0], [30.0]]), [0, 1], [])
    padded, transform, _, shape = padded_raster(str(path), 1)
    hourly = {"values": np.column_stack([10 + np.arange(10.0), 30 + np.arange(10.0)]), "location_ids": np.array([1, 2]),
              "start": pd.Timestamp("2025-01-01 00:00")}
    ref_xy = np.array([[5.0, size * 10 - 5.0], [size * 10 - 5.0, 5.0]])
    return predict_surface(Net(), scalers, padded, shape, transform, hourly, [1, 2], ref_xy,
                           pd.Timestamp("2025-01-01 06:00"), cfg)[0]


def test_a_big_grid_reports_progress_and_time_remaining(cfg, tmp_path, capsys):
    pred = _predict(cfg, tmp_path, 250)                          # 62,500 cells: above the 50,000 threshold
    out = capsys.readouterr().out
    assert pred.shape == (250, 250) and np.isfinite(pred).all()
    assert "%" in out and "remaining" in out and "elapsed" in out


def test_a_small_grid_stays_quiet(cfg, tmp_path, capsys):
    _predict(cfg, tmp_path, 100)                                 # 10,000 cells
    assert capsys.readouterr().out == ""


def test_durations_are_printed_readably():
    from inference.cell_inference import format_duration
    assert format_duration(30) == "30 s" and format_duration(600) == "10 min" and format_duration(36000) == "10.0 h"
