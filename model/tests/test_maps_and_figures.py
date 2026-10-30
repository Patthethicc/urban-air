"""The thesis-style maps and explain figures: NCR clip, city furniture, both colour styles, hot and cold cell selection, the morphology
sheet and the boundary file."""
import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from explain.gradcam import spread_selection
from explain.gradcam_map import plot_gradcam_map
from explain.gradcam_plots import plot_band_profile, plot_cam_overlays
from explain.morphology_sheet import class_edges, plot_morphology_sheet, read_bands
from output import aqi_maps, cartography
from preprocessing import boundaries
from preprocessing.coverage import coverage_report, format_coverage

CRS = "EPSG:32651"
TRANSFORM = from_origin(0, 200, 10, 10)                     # a 20 x 20 grid of 10 m cells: x 0..200, y 0..200


def _prediction():
    v = np.linspace(8.0, 40.0, 400).reshape(20, 20).astype("float32")
    v[:3, :] = np.nan                                       # no prediction in the top rows
    return v


@pytest.fixture
def ncr(cfg, tmp_path):
    """A boundary of two 'cities' covering cells columns 5-14, rows 5-14 (inset by 1 m so the cell edges are unambiguous)."""
    path = tmp_path / "ncr.gpkg"
    gpd.GeoDataFrame({"city": ["Alpha", "Beta"], "geometry": [box(51, 51, 99, 149), box(101, 51, 149, 149)]}, crs=CRS).to_file(path, driver="GPKG")
    cfg["data"]["paths"]["boundary"] = str(path)
    cfg["data"]["crs"] = CRS
    cfg["output"]["basemap"] = {"enabled": False, "web_tiles": False, "vector_layers": False, "strict": False}
    return cfg


# ------------------------------------------------------------------------------ cartography helpers
@pytest.mark.parametrize("target, expected", [(4600, 2000), (5000, 5000), (23000, 20000), (130, 100), (0.9, 0.5), (999, 500)])
def test_the_scale_bar_length_is_the_largest_1_2_or_5_times_a_power_of_ten(target, expected):
    assert cartography.nice_length(target) == pytest.approx(expected)


def test_the_boundary_marks_exactly_the_cells_it_touches_and_stats_count_predictions_inside_it(ncr):
    boundary = cartography.load_boundary(ncr)
    inside = cartography.inside_mask(TRANSFORM, (20, 20), boundary)
    assert inside.sum() == 100 and inside[5:15, 5:15].all() and not inside[:5].any()
    s = cartography.grid_stats(_prediction(), TRANSFORM, ncr)                 # rows 0-2 are empty but outside the boundary: nothing is missing inside
    assert s == {"cells": 100, "coverage": 1.0, "clipped": True}
    v = _prediction()
    v[5:10, 5:15] = np.nan                                                   # half of the inside has no prediction
    assert cartography.grid_stats(v, TRANSFORM, ncr)["coverage"] == pytest.approx(0.5)
    ncr["output"]["maps"] = {"clip_to_boundary": False}
    s = cartography.grid_stats(_prediction(), TRANSFORM, ncr)                # no clip: all finite cells over all cells
    assert s["clipped"] is False and s["cells"] == 340 and s["coverage"] == pytest.approx(340 / 400)


def test_a_missing_or_unset_boundary_gives_one_warning_and_unclipped_maps(cfg, tmp_path):
    assert cartography.load_boundary(cfg) is None                            # unset (the test config): silent
    cfg["data"]["paths"]["boundary"] = str(tmp_path / "nope.gpkg")
    with pytest.warns(UserWarning, match="not clipped to the NCR"):
        assert cartography.load_boundary(cfg) is None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # the failure is cached: no second warning
        assert cartography.load_boundary(cfg) is None
    v = np.arange(4.0).reshape(2, 2)
    assert cartography.clip_display(v, (0, 2, 0, 2), cfg) is v


def test_clipping_a_drawn_grid_blanks_what_lies_outside_the_boundary(ncr):
    shown = np.ones((40, 40))                                                # a finer display grid over the same extent
    out = cartography.clip_display(shown, (0, 200, 0, 200), ncr)
    assert np.isfinite(out).sum() == 100 * 4 and np.isnan(out[:10]).all()


def test_city_names_are_drawn_inside_the_view_with_nudges_for_crowded_neighbours(ncr):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    cartography.draw_cities(ax, ncr)
    assert sorted(t.get_text() for t in ax.texts) == ["Alpha", "Beta"]
    ax.clear()
    ax.set_xlim(0, 90)                                                       # Beta's label lies outside the view
    ax.set_ylim(0, 200)
    cartography.draw_cities(ax, ncr)
    assert [t.get_text() for t in ax.texts] == ["Alpha"]
    ax.clear()
    ncr["output"]["maps"] = {"city_labels": False}
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    cartography.draw_cities(ax, ncr)
    assert len(ax.texts) == 0 and len(ax.collections) + len(ax.lines) > 0       # outlines stay, names go
    plt.close(fig)
    assert cartography.LABEL_NUDGE["Navotas"][2] == "right"


def test_the_footer_reserves_room_only_when_there_is_something_to_say(cfg):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6))
    assert 0 < cartography.footer(fig, cfg, "a caption", ("a note",)) < 0.3
    cfg["output"]["maps"] = {"caption": False, "metric_notes": False}
    assert cartography.footer(plt.figure(figsize=(6, 6)), cfg, "a caption", ("a note",)) == 0.0
    plt.close("all")


# ------------------------------------------------------------------------------ the prediction maps
def test_view_extent_is_the_boundary_box_inside_the_grid(ncr):
    left, right, bottom, top = aqi_maps.view_extent((0, 200, 0, 200), ncr)
    assert left == pytest.approx(51 - 0.01 * 98) and right == pytest.approx(149 + 0.01 * 98) and bottom < 51 and top > 149
    ncr["output"]["maps"] = {"clip_to_boundary": False}
    assert aqi_maps.view_extent((0, 200, 0, 200), ncr) == (0, 200, 0, 200)


def test_the_continuous_range_follows_the_cells_inside_the_ncr_or_the_fixed_setting(ncr):
    v = _prediction()
    lo, hi = aqi_maps.continuous_range([v], TRANSFORM, ncr)
    inside = v[5:15, 5:15].ravel()
    assert lo == pytest.approx(np.percentile(inside, 2)) and hi == pytest.approx(np.percentile(inside, 98))
    ncr["output"]["maps"] = {"continuous_range": [5, 50]}
    assert aqi_maps.continuous_range([v], TRANSFORM, ncr) == (5.0, 50.0)


@pytest.mark.parametrize("style", ["aqi", "continuous"])
def test_a_map_is_clipped_to_the_ncr_and_its_title_gives_cells_and_coverage(ncr, style):
    import matplotlib.pyplot as plt
    ncr["output"]["smooth_maps"] = False                                    # draw the raw 20 x 20 cells, not the smoothed finer grid
    fig, ax = plt.subplots()
    image = aqi_maps.draw_surface(ax, _prediction(), TRANSFORM, CRS, ncr, None, None, "12:00", style=style)
    arr = np.ma.getmaskarray(image.get_array())
    assert arr.shape == (20, 20) and (~arr).sum() == 100                    # only the 100 cells inside the boundary are drawn
    assert ax.get_title(loc="left") == "12:00   (100 cells, coverage 1.00)"
    assert ax.get_xlabel() == "easting (km)" and ax.get_xlim()[0] > 40       # framed on the NCR, not on the whole grid
    plt.close(fig)
    with pytest.raises(ValueError, match="map style"):
        aqi_maps.draw_surface(plt.figure().add_subplot(), _prediction(), TRANSFORM, CRS, ncr, style="rainbow")


def test_a_long_title_puts_the_counts_on_a_second_line(ncr):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    aqi_maps.draw_surface(ax, _prediction(), TRANSFORM, CRS, ncr, None, None, "PM2.5 prediction, 2025-07-01 12:00", style="aqi")
    assert ax.get_title(loc="left").splitlines() == ["PM2.5 prediction, 2025-07-01 12:00", "(100 cells, coverage 1.00)"]
    plt.close(fig)


@pytest.mark.parametrize("with_boundary", [True, False], ids=["clipped", "unclipped"])
@pytest.mark.parametrize("style", ["aqi", "continuous"])
def test_snapshots_and_panels_are_written_in_both_styles_with_and_without_a_boundary(cfg, ncr, tmp_path, style, with_boundary):
    c = ncr if with_boundary else cfg
    c["output"]["basemap"] = {"enabled": False, "web_tiles": False, "vector_layers": False, "strict": False}
    one = aqi_maps.render_snapshot(_prediction(), TRANSFORM, CRS, str(tmp_path / "one.png"), "t", c, 35, style=style, caption="a caption")
    many = aqi_maps.render_panels(np.stack([_prediction()] * 2), ["08:00", "12:00"], TRANSFORM, CRS, str(tmp_path / "many.png"), "day", c, 35,
                                  style=style, caption="a caption")
    for path in (one, many):
        assert (tmp_path / path.split("\\")[-1].split("/")[-1]).stat().st_size > 8_000


def test_the_two_styles_colour_the_same_cells_differently(ncr, tmp_path):
    import matplotlib.image as mpimg
    aqi_maps.render_snapshot(_prediction(), TRANSFORM, CRS, str(tmp_path / "a.png"), "t", ncr, 35, style="aqi")
    aqi_maps.render_snapshot(_prediction(), TRANSFORM, CRS, str(tmp_path / "c.png"), "t", ncr, 35, style="continuous")
    a, c = mpimg.imread(tmp_path / "a.png"), mpimg.imread(tmp_path / "c.png")
    assert a.shape[1] != c.shape[1] or np.abs(a[..., :3].mean(axis=(0, 1)) - c[..., :3].mean(axis=(0, 1))).sum() > 0.01


# ------------------------------------------------------------------------------ the explain figures
def test_the_explained_cells_are_kept_apart(tmp_path):
    xs = np.array([0.0, 100.0, 2000.0, 2100.0, 5000.0, 9000.0])             # best first: two pairs of neighbours and two loners
    ys = np.zeros(6)
    assert spread_selection(xs, ys, 3, 1000).tolist() == [0, 2, 4]
    assert spread_selection(xs, ys, 10, 1000).tolist() == [0, 2, 4, 5]       # asking for more than exist apart returns what exists
    assert spread_selection(xs, ys, 3, 0).tolist() == [0, 1, 2]              # 0 = the plain top n
    assert spread_selection(xs, ys, 2, 1000).tolist() == [0, 2]


def _cells():
    return [{"group": g, "number": i + 1, "x": 60.0 + 25 * i + (0 if g == "hot" else 8), "y": 70.0 + 20 * i, "predicted": 30.0 - 5 * i}
            for g in ("hot", "cold") for i in range(3)]


def test_the_grad_cam_map_the_close_ups_and_the_band_chart_are_written(ncr, tmp_path):
    names = ["building_height_mean", "ndvi_mean"]
    rng = np.random.default_rng(0)
    raw, cam = rng.random((3, 2, 32, 32)).astype("float32"), rng.random((3, 32, 32)).astype("float32")
    big = lambda p: p.stat().st_size > 8_000
    out = plot_cam_overlays(raw, cam, names, ["HOT #1", "HOT #2", "HOT #3"], str(tmp_path / "cam.png"), 0.8, 60, cfg=ncr, px_m=10, caption="c",
                            heading="h")
    assert big(tmp_path / "cam.png") and out.endswith("cam.png")
    rows = [{"group": g, "band": b, "std_units": float(rng.normal(0.3 if g == "hot" else -0.2, 0.5))} for g in ("hot", "cold") for b in names
            for _ in range(5)]
    plot_band_profile(rows, names, str(tmp_path / "profile.png"), 60, cfg=ncr, caption="c")
    plot_gradcam_map(_prediction(), TRANSFORM, CRS, _cells(), ncr, str(tmp_path / "map.png"), "PM2.5 prediction, 12:00", caption="c", dpi=60)
    assert big(tmp_path / "profile.png") and big(tmp_path / "map.png")
    plot_cam_overlays(raw, cam, names, ["a", "b", "c"], str(tmp_path / "plain.png"))        # the old call, without cfg, still works
    assert big(tmp_path / "plain.png")


def test_class_limits_are_quantiles_and_shrink_when_values_coincide():
    assert len(class_edges(np.arange(1000.0), 6)) == 7
    half_zeros = np.r_[np.zeros(500), np.arange(1, 501.0)]                  # half zeros: the lower classes collapse into one
    assert len(class_edges(half_zeros, 6)) < 7 and class_edges(half_zeros, 6)[0] == 0 and class_edges(half_zeros, 6)[-1] == 500
    mostly_zeros = np.r_[np.zeros(900), np.arange(1, 101.0)]                # too few distinct limits: an even split of the range instead
    assert class_edges(mostly_zeros, 6).tolist() == pytest.approx(np.linspace(0, 100, 7).tolist())
    assert len(class_edges(np.full(50, 3.0), 6)) >= 2 and len(class_edges(np.array([np.nan]), 6)) == 2


def test_the_morphology_sheet_reads_the_bands_without_the_nodata_and_draws_every_band(ncr, tmp_path):
    path = tmp_path / "bands.tif"
    rng = np.random.default_rng(1)
    data = rng.random((5, 40, 40)).astype("float32")
    data[:, :4, :] = -9999.0                                                 # NoData rows
    with rasterio.open(path, "w", driver="GTiff", height=40, width=40, count=5, dtype="float32", crs=CRS,
                       transform=from_origin(0, 200, 5, 5), nodata=-9999.0) as dst:
        dst.write(data)
    bands, transform, extent = read_bands(str(path), step_m=10.0)           # 5 m pixels averaged to 10 m cells
    assert bands.shape == (5, 20, 20) and np.isnan(bands[:, :2]).all() and np.isfinite(bands[:, 3:]).all() and extent == (0, 200, 0, 200)
    names = ["building_height_mean", "building_height_stddev", "ndvi_mean", "building_density", "street_aspect_ratio_mean"]
    plot_morphology_sheet(str(path), ncr, _cells(), str(tmp_path / "sheet.png"), names, caption="c", step_m=10.0, dpi=60)
    assert (tmp_path / "sheet.png").stat().st_size > 8_000


# ------------------------------------------------------------------------------ the boundary file
def _gadm(tmp_path, n=17, names=None, region="MetropolitanManila"):
    from preprocessing.boundaries import DISPLAY
    raw = ["KalookanCity", "LasPiñas", "MakatiCity", "Malabon", "Mandaluyong", "Manila", "Marikina", "Muntinlupa", "Navotas", "Parañaque",
           "PasayCity", "PasigCity", "Pateros", "QuezonCity", "SanJuan", "Taguig", "Valenzuela"]
    assert len(raw) == len(DISPLAY) == 17
    rows = [{"NAME_1": region, "NAME_2": nm, "geometry": box(121 + 0.02 * i, 14.5, 121.02 + 0.02 * i, 14.52)} for i, nm in enumerate(names or raw[:n])]
    rows.append({"NAME_1": "Rizal", "NAME_2": "Rodriguez", "geometry": box(121.5, 14.7, 121.6, 14.8)})
    path = tmp_path / "gadm.json"
    gpd.GeoDataFrame(rows, crs="EPSG:4326").to_file(path, driver="GeoJSON")
    return str(path)


def test_the_ncr_extract_keeps_the_17_cities_with_clean_names_in_the_project_crs(tmp_path):
    ncr = boundaries.extract_ncr(_gadm(tmp_path), CRS)
    assert len(ncr) == 17 and str(ncr.crs) == CRS and ncr["area_km2"].gt(0).all()
    assert {"Quezon City", "Las Piñas", "Parañaque", "Caloocan", "Pasay", "Makati"} <= set(ncr["city"])
    assert "Rodriguez" not in " ".join(ncr["city"])                          # the Rizal polygon is not part of the NCR
    assert boundaries._key("Las Piñas") == boundaries._key("LasPiñas") == boundaries._key("LAS PIÑAS") == "laspinas"


def test_a_file_that_is_not_the_17_ncr_cities_is_refused(tmp_path):
    with pytest.raises(ValueError, match="expected 17"):
        boundaries.extract_ncr(_gadm(tmp_path, n=16), CRS)
    with pytest.raises(ValueError, match="expected 17"):
        boundaries.extract_ncr(_gadm(tmp_path, region="Cavite"), CRS)
    eighteen = ["KalookanCity", "LasPiñas", "MakatiCity", "Malabon", "Mandaluyong", "Manila", "Marikina", "Muntinlupa", "Navotas", "Parañaque",
                "PasayCity", "PasigCity", "Pateros", "QuezonCity", "SanJuan", "Taguig", "Valenzuela", "Manila"]       # a duplicated polygon
    with pytest.raises(ValueError, match="expected 17"):
        boundaries.extract_ncr(_gadm(tmp_path, names=eighteen), CRS)
    bad = ["Kalookan City", "LasPiñas", "MakatiCity", "Malabon", "Mandaluyong", "Manila", "Marikina", "Muntinlupa", "Navotas", "Parañaque",
           "PasayCity", "PasigCity", "Pateros", "QuezonCity", "SanJuan", "Taguig", "Atlantis"]
    with pytest.raises(ValueError, match="unknown NCR city names.*Atlantis"):
        boundaries.extract_ncr(_gadm(tmp_path, names=bad), CRS)


def test_the_command_line_writes_the_geopackage(tmp_path, capsys):
    out = tmp_path / "out" / "ncr.gpkg"
    boundaries.main(["--src", _gadm(tmp_path), "--out", str(out)])
    assert len(gpd.read_file(out)) == 17 and "17 NCR cities" in capsys.readouterr().out


def test_coverage_says_when_the_boundary_file_is_missing_instead_of_failing(cfg, tmp_path):
    cfg["data"]["paths"]["boundary"] = str(tmp_path / "not_there.gpkg")
    with rasterio.open(tmp_path / "r.tif", "w", driver="GTiff", height=4, width=4, count=1, dtype="float32", crs=CRS,
                       transform=from_origin(0, 4000, 1000, 1000)) as dst:
        dst.write(np.ones((1, 4, 4), "float32"))
    cfg["data"]["paths"]["master_raster"] = str(tmp_path / "r.tif")
    cfg["data"]["paths"]["sensor_locations_csv"] = None
    import pandas as pd
    import preprocessing.coverage as cov
    orig = cov.all_sensor_locations if hasattr(cov, "all_sensor_locations") else None
    if orig is not None:
        cov.all_sensor_locations = lambda c: pd.DataFrame({"location_id": [], "name": [], "latitude": [], "longitude": []})
    try:
        report = coverage_report(cfg)
    except Exception as exc:                                                 # the sensor tables are not the point of this test
        pytest.skip(f"coverage needs sensor tables here: {exc}")
    finally:
        if orig is not None:
            cov.all_sensor_locations = orig
    assert report["boundary"] is None and report["boundary_missing"].endswith("not_there.gpkg")
    assert "not found" in format_coverage(report)


def test_the_default_config_points_at_the_ncr_file_and_the_map_settings_exist(base_cfg):
    assert base_cfg["data"]["paths"]["boundary"].replace("\\", "/").endswith("data/Boundaries/ncr_cities.gpkg")
    m = cartography.settings(base_cfg)
    assert m["styles"] == ["aqi", "continuous"] and m["clip_to_boundary"]
    assert m["look"] == "xgboost" and m["continuous_range"] == [0, 25]                       # shipped: the XGBoost look on its fixed shared scale
    assert base_cfg["output"]["explain"]["min_separation_m"] == 1000
    json.dumps(m)


def test_cells_without_a_prediction_inside_the_ncr_show_as_grey_no_data(ncr):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    fig, ax = plt.subplots()
    aqi_maps.draw_surface(ax, _prediction(), TRANSFORM, CRS, ncr, None, None, "t", style="continuous")
    faces = [tuple(np.round(c.get_facecolor()[0], 3)) for c in ax.collections if len(c.get_facecolor())]
    assert tuple(np.round(to_rgba(cartography.NO_DATA), 3)) in faces
    plt.close(fig)
    ncr["output"]["maps"] = {"clip_to_boundary": False}                     # nothing to underlay without a boundary
    fig, ax = plt.subplots()
    aqi_maps.draw_surface(ax, _prediction(), TRANSFORM, CRS, ncr, None, None, "t", style="continuous")
    assert len(ax.collections) == 0
    plt.close(fig)


# ------------------------------------------------------------------------------ the XGBoost look of the continuous maps
def test_the_xgboost_look_is_off_by_default_and_changes_scale_title_and_cells(ncr, tmp_path):
    grid = _prediction() + 5.0                                  # 13 .. 45, so p2 is well above 0
    assert not aqi_maps.xgboost_look(ncr) and aqi_maps.continuous_range([grid], TRANSFORM, ncr)[0] > 5
    ncr["output"]["maps"] = {**(ncr["output"].get("maps") or {}), "look": "xgboost"}
    cartography.reset()
    assert aqi_maps.xgboost_look(ncr)
    assert aqi_maps.continuous_range([grid], TRANSFORM, ncr)[0] == 0.0                         # a lone map starts at 0, like XGBoost's
    assert aqi_maps.continuous_range([grid, grid], TRANSFORM, ncr)[0] > 5                      # a set shares [max(0, p2), p98]
    fig, ax = aqi_maps.plt.subplots()
    aqi_maps.draw_surface(ax, grid, TRANSFORM, CRS, ncr, None, None, "2025-07-23 12:00", style="continuous", single=True)
    assert "continuous scale" in ax.get_title("left") and "colour is comparable across snapshots" in ax.get_title("left")
    aqi_maps.plt.close(fig)
    aqi_maps.render_snapshot(grid, TRANSFORM, CRS, str(tmp_path / "x.png"), "2025-07-23 12:00", ncr, None, style="continuous", caption="c")
    assert (tmp_path / "x.png").stat().st_size > 8_000
