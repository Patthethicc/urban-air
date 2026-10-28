"""Hot and cold cells of the high and low end of each morphology band over the NCR (explain/feature_groups.py): the groups come from the
raster, only the picks inside a group come from the prediction."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from explain import feature_groups as fg
from explain.feature_groups_plots import plot_group_comparison, plot_group_map

CRS = "EPSG:32651"
TRANSFORM = from_origin(0, 200, 10, 10)                       # a 20 x 20 grid of 10 m cells: x 0..200, y 0..200
NAMES = ["building_height_mean", "ndvi_mean"]


@pytest.fixture
def ncr(cfg, tmp_path):
    path = tmp_path / "ncr.gpkg"
    gpd.GeoDataFrame({"city": ["Alpha", "Beta"], "geometry": [box(1, 1, 99, 199), box(101, 1, 199, 199)]}, crs=CRS).to_file(path, driver="GPKG")
    cfg["data"]["paths"]["boundary"] = str(path)
    cfg["data"]["crs"] = CRS
    cfg["output"]["basemap"] = {"enabled": False, "web_tiles": False, "vector_layers": False, "strict": False}
    return cfg


def _raster(tmp_path, band0, band1, nodata=-9999.0):
    path = tmp_path / "bands.tif"
    with rasterio.open(path, "w", driver="GTiff", height=20, width=20, count=2, dtype="float32", crs=CRS, transform=TRANSFORM,
                       nodata=nodata) as dst:
        dst.write(np.stack([band0, band1]).astype("float32"))
    return str(path)


def _gradient():
    col = np.tile(np.arange(20.0), (20, 1))
    return col                                                 # band value = the column: the high end is the east


# ------------------------------------------------------------------------------ the pieces
def test_the_patch_mean_ignores_nodata_and_drops_patches_that_are_mostly_empty(tmp_path):
    band = np.arange(400.0).reshape(20, 20)
    band[0:3, 0:3] = -9999.0                                   # an empty corner
    path = _raster(tmp_path, band, np.ones((20, 20)))
    grids, transform, extent = fg.patch_mean_grids(path, patch_m=30.0, step_m=10.0)        # 3 x 3 cell window
    assert grids.shape == (2, 20, 20) and extent == (0, 200, 0, 200)
    assert grids[0, 10, 10] == pytest.approx(band[9:12, 9:12].mean())                       # brute force
    assert np.isnan(grids[0, 1, 1])                                                         # window holds only nodata
    assert grids[0, 3, 3] == pytest.approx(band[2:5, 2:5][band[2:5, 2:5] > 0].mean())      # nodata left out of the mean, not counted as 0
    assert grids[1, 10, 10] == pytest.approx(1.0) and np.isnan(grids[0, 0, 0])


def test_the_grid_is_sampled_under_each_coordinate_and_is_nan_outside_it():
    grid = np.arange(400.0).reshape(20, 20)
    got = fg.sample_grid(grid, TRANSFORM, [5.0, 195.0, -5.0, 105.0], [195.0, 5.0, 100.0, 95.0])
    assert got[0] == grid[0, 0] and got[1] == grid[19, 19] and np.isnan(got[2]) and got[3] == grid[10, 10]


def test_limits_are_percentiles_and_a_band_that_cannot_be_split_has_none():
    v = np.r_[np.arange(1001.0), np.nan]
    assert fg.thresholds(v, 90, 10) == pytest.approx((100.0, 900.0))
    assert fg.thresholds(np.full(50, 3.0), 90, 10) is None and fg.thresholds(np.array([np.nan]), 90, 10) is None
    high, low = fg.membership(np.array([1.0, 5.0, 9.0, np.nan, 9.0]), 1.0, 9.0)
    assert high.tolist() == [False, False, True, False, True] and low.tolist() == [True, False, False, False, False]


def test_hot_and_cold_are_the_extreme_predictions_inside_the_group_only():
    pred = np.array([50.0, 10.0, 40.0, 20.0, 30.0, 5.0, 60.0, 15.0])
    xs, ys = np.arange(8) * 5000.0, np.zeros(8)
    member = np.array([False, True, True, True, True, True, False, True])    # cells 0 and 6 are the global hottest, but not in the group
    hot, cold = fg.pick_hot_cold(pred, xs, ys, member, 2, 2, 1000)
    assert hot.tolist() == [2, 4] and cold.tolist() == [5, 1]                # 40, 30 hottest of the group; 5, 10 coldest


def test_picks_are_spread_apart_and_a_small_group_never_lists_a_cell_twice():
    pred = np.array([9.0, 8.0, 7.0, 1.0])
    xs, ys = np.array([0.0, 100.0, 5000.0, 6000.0]), np.zeros(4)
    hot, cold = fg.pick_hot_cold(pred, xs, ys, np.ones(4, bool), 2, 2, 1000)
    assert hot.tolist() == [0, 2]                                            # cell 1 is within 1 km of cell 0
    assert cold.tolist() == [3, 1] and not set(hot) & set(cold)
    hot, cold = fg.pick_hot_cold(pred, xs, ys, np.array([True, False, False, True]), 2, 2, 0)
    assert hot.tolist() == [0, 3] and cold.size == 0                         # two cells only: the hot picks keep them


def test_the_city_table_gives_each_citys_share_of_the_groups_and_their_mean_prediction():
    pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    city = np.array([1, 1, 1, 2, 2, 0])
    memberships = {("b", "zone"): (np.array([0, 0, 1, 1, 1, 1], bool), np.array([1, 0, 0, 0, 0, 0], bool))}
    t = fg.city_table(pred, city, ["Alpha", "Beta"], memberships).set_index("city")
    assert t.loc["Alpha", "cells"] == 3 and t.loc["Alpha", "share_high"] == pytest.approx(1 / 3) and t.loc["Alpha", "mean_prediction_high"] == 30.0
    assert t.loc["Beta", "share_high"] == 1.0 and t.loc["Beta", "mean_prediction_high"] == 45.0 and np.isnan(t.loc["Beta", "mean_prediction_low"])
    assert t.loc["Alpha", "mean_prediction_low"] == 10.0 and len(t) == 2      # the cell outside every city is in no row


def test_cities_are_numbered_on_the_grid(ncr):
    grid, names = fg.city_index(TRANSFORM, (20, 20), __import__("output.cartography", fromlist=["x"]).active_boundary(ncr))
    assert names == ["Alpha", "Beta"] and grid[10, 2] == 1 and grid[10, 17] == 2


# ------------------------------------------------------------------------------ the whole pass
def _prediction():
    col = np.tile(np.arange(20.0), (20, 1))
    row = np.arange(20.0)[:, None]
    return (10 + 0.5 * (19 - col) + 0.2 * row).astype("float32")             # high in the west: the opposite of band 0


def _explain_cells(calls):
    def explain(xs, ys):
        calls.append((np.array(xs), np.array(ys)))
        n = len(xs)
        keep = np.ones(n, bool)
        keep[-1] = False if n > 5 else True                                  # one cell without a history
        n_kept = int(keep.sum())
        rng = np.random.default_rng(0)
        return rng.random((n_kept, 2, 4, 4)).astype("float32"), rng.random((n_kept, 4, 4)).astype("float32"), np.linspace(20, 30, n_kept), keep
    return explain


def test_groups_come_from_the_raster_and_the_prediction_only_picks_inside_them(ncr, tmp_path):
    ncr["output"]["explain"].update({"top_n": 2, "bottom_n": 2, "min_separation_m": 0})
    ncr["output"]["explain"]["feature_groups"] = {"levels": [{"name": "zone", "high_percentile": 75, "low_percentile": 25},
                                                             {"name": "extreme", "high_percentile": 95, "low_percentile": 5}],
                                                  "feature_step_m": 10}
    path = _raster(tmp_path, _gradient(), np.tile(np.arange(20.0)[:, None], (1, 20)))    # band 0 rises east, band 1 rises north-south
    pred_map = _prediction()
    ok = np.ones((20, 20), bool)
    calls = []
    summary = fg.run(ncr, pred_map=pred_map, transform=TRANSFORM, ok=ok, names=NAMES, raster_path=path, explain_cells=_explain_cells(calls),
                     out_dir=tmp_path, ts=pd.Timestamp("2025-07-01 12:00"), caption="c", dpi=50, px=3, res=10.0)
    folder = tmp_path / "feature_groups"
    for f in ("feature_group_summary.csv", "feature_group_cells.csv", "feature_group_attention.csv", "feature_group_cities.csv",
              "feature_group_comparison.png", "map_building_height_mean_zone.png", "map_ndvi_mean_extreme.png",
              "gradcam_building_height_mean_zone_high.png", "gradcam_ndvi_mean_extreme_low.png"):
        assert (folder / f).stat().st_size > 500, f
    cells = pd.read_csv(folder / "feature_group_cells.csv")
    zone = cells[(cells["group_band"] == "building_height_mean") & (cells["level"] == "zone")]
    high_hot = zone[(zone["side"] == "high") & (zone["kind"] == "hot")]
    limits = pd.read_csv(folder / "feature_group_summary.csv").set_index(["band", "level", "side"])
    assert (high_hot["x"] > 100).all()                                       # the high end of band 0 is the east, although the prediction is high in the west
    assert limits.loc[("building_height_mean", "zone", "high"), "cells"] > 0 and limits.loc[("building_height_mean", "zone", "low"), "cells"] > 0
    high_pred = limits.loc[("building_height_mean", "zone", "high"), "mean_prediction"]
    low_pred = limits.loc[("building_height_mean", "zone", "low"), "mean_prediction"]
    assert high_pred < low_pred                                              # east is cleaner than west in this synthetic surface
    assert summary["reading"].startswith("association") and summary["hot_per_group"] == 2 and not summary["skipped"]
    assert len(calls) == 8                                                   # 2 bands x 2 levels x (high, low)
    assert set(zone["kind"]) <= {"hot", "cold"} and (zone["number"] >= 1).all()


def test_a_band_that_is_constant_is_skipped_and_said_so(ncr, tmp_path):
    ncr["output"]["explain"]["feature_groups"] = {"feature_step_m": 10}
    path = _raster(tmp_path, _gradient(), np.full((20, 20), 4.0))
    summary = fg.run(ncr, pred_map=_prediction(), transform=TRANSFORM, ok=np.ones((20, 20), bool), names=NAMES, raster_path=path,
                     explain_cells=_explain_cells([]), out_dir=tmp_path, ts=pd.Timestamp("2025-07-01 12:00"), caption=None, dpi=50, px=3, res=10.0)
    assert {s["band"] for s in summary["skipped"]} == {"ndvi_mean"} and len(summary["skipped"]) == 2


def test_without_a_boundary_the_pass_still_runs_and_skips_the_city_table(cfg, tmp_path):
    cfg["data"]["paths"]["boundary"] = None
    cfg["output"]["explain"]["feature_groups"] = {"feature_step_m": 10, "levels": [{"name": "zone", "high_percentile": 75, "low_percentile": 25}]}
    path = _raster(tmp_path, _gradient(), np.tile(np.arange(20.0)[:, None], (1, 20)))
    fg.run(cfg, pred_map=_prediction(), transform=TRANSFORM, ok=np.ones((20, 20), bool), names=NAMES, raster_path=path,
           explain_cells=_explain_cells([]), out_dir=tmp_path, ts=pd.Timestamp("2025-07-01 12:00"), caption=None, dpi=50, px=3, res=10.0)
    assert (tmp_path / "feature_groups" / "feature_group_summary.csv").exists() and not (tmp_path / "feature_groups" / "feature_group_cities.csv").exists()


def test_the_settings_default_to_on_with_a_zone_and_an_extreme_level(cfg):
    s = fg.settings(cfg)
    assert s["enabled"] and [l["name"] for l in s["levels"]] == ["zone", "extreme"] and s["levels"][1]["high_percentile"] == 99.9


def test_the_group_map_and_the_comparison_chart_are_written(ncr, tmp_path):
    grid = _gradient()
    picks = [{"side": s, "kind": k, "number": 1, "x": 50.0 + 40 * (s == "high"), "y": 80.0 + 30 * (k == "hot")}
             for s in ("high", "low") for k in ("hot", "cold")]
    plot_group_map(grid, (0, 200, 0, 200), TRANSFORM, "ndvi_mean", "zone", (4.0, 15.0), picks, ncr, str(tmp_path / "m.png"), caption="c", dpi=60)
    table = pd.DataFrame([{"band": b, "level": lv, "side": sd, "cells": 50, "mean_prediction": 20.0 + i, "sd_prediction": 3.0}
                          for i, (b, lv, sd) in enumerate((b, lv, sd) for b in NAMES for lv in ("zone", "extreme") for sd in ("high", "low"))])
    plot_group_comparison(table, 21.0, str(tmp_path / "c.png"), cfg=ncr, caption="c", dpi=60)
    assert (tmp_path / "m.png").stat().st_size > 8_000 and (tmp_path / "c.png").stat().st_size > 8_000
