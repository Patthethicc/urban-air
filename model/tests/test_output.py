import importlib.util

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from output.aqi_maps import categorize, render_panels, render_snapshot
from output.volume import daily_mean, expand_dates, group_by_date, summarize_date, write_volume
from uncertainty.geocp import GeoCP
from uncertainty.geocp_surface import interval_layers

TRANSFORM = from_origin(0, 100, 10, 10)
HAS_BASEMAP = importlib.util.find_spec("contextily") is not None


def test_aqi_category_edges_follow_the_philippine_scale(cfg):
    values = np.array([10, 25, 25.1, 35, 35.1, 45, 45.1, 55, 56, np.nan])
    got = categorize(values, cfg)
    assert got[:9].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4] and np.isnan(got[9])    # 35.0 is Fair, 35.1 is unhealthy for sensitive groups


def test_the_highlight_threshold_is_the_text_limit_and_the_guideline_is_50(base_cfg):
    assert base_cfg["output"]["highlight_threshold"] == 35 and base_cfg["output"]["daily_guideline"] == 50
    assert base_cfg["output"]["aqi"]["bins"][2]["upper"] == 45                      # unhealthy for sensitive groups 35.1-45


def test_the_four_daily_snapshots_are_08_12_17_21(base_cfg):
    stamps = expand_dates(["2025-07-01"], base_cfg["inference"]["snapshot_hours"])
    assert [t.strftime("%H:%M") for t in stamps] == ["08:00", "12:00", "17:00", "21:00"]
    grouped = group_by_date(stamps + [pd.Timestamp("2025-07-02 12:00")])
    assert [len(v) for v in grouped.values()] == [4, 1]


def test_daily_mean_needs_every_snapshot():
    stack = np.array([[[10.0, 20.0]], [[30.0, np.nan]]])
    out = daily_mean(stack)
    assert out[0, 0] == 20.0 and np.isnan(out[0, 1])


def test_volume_has_one_band_per_snapshot_labelled_by_hour(tmp_path):
    stack = np.arange(24, dtype="float32").reshape(4, 2, 3)
    path = write_volume(stack, TRANSFORM, "EPSG:32651", str(tmp_path / "v.tif"), ["08:00", "12:00", "17:00", "21:00"])
    with rasterio.open(path) as r:
        assert r.count == 4 and r.descriptions == ("08:00", "12:00", "17:00", "21:00")
        assert np.array_equal(r.read(3), stack[2])


def test_summary_counts_cells_above_the_limits(cfg):
    pred = np.array([[[10.0, 40.0, 60.0, np.nan]]] * 2)
    pred[1] = [[30.0, 36.0, 80.0, np.nan]]
    width = np.full_like(pred, 13.0)
    stamps = [pd.Timestamp("2025-07-01 08:00"), pd.Timestamp("2025-07-01 12:00")]
    s = summarize_date(stamps, pred, width, cfg)
    assert s["snapshots"][0]["cells_above_35"] == 2 and s["snapshots"][1]["cells_above_35"] == 2
    assert s["snapshots"][0]["cells"] == 3
    assert s["daily_mean"]["cells_above_50"] == 1 and s["daily_mean"]["cells"] == 3        # means: 20, 38, 70


def _prediction():
    p = np.linspace(10, 70, 100).reshape(10, 10)
    p[0, 0] = np.nan
    return p


def test_maps_are_written_without_a_basemap(cfg, tmp_path):
    cfg["output"]["basemap"] = "none"
    sensors = pd.DataFrame({"x": [15.0, 85.0], "y": [85.0, 15.0]})
    one = render_snapshot(_prediction(), TRANSFORM, "EPSG:32651", str(tmp_path / "one.png"), "t", cfg, 35, sensors)
    many = render_panels(np.stack([_prediction()] * 4), ["08:00", "12:00", "17:00", "21:00"], TRANSFORM, "EPSG:32651",
                         str(tmp_path / "many.png"), "day", cfg, 35, sensors)
    assert (tmp_path / "one.png").stat().st_size > 5000 and (tmp_path / "many.png").stat().st_size > 5000
    assert one.endswith("one.png") and many.endswith("many.png")


@pytest.mark.skipif(HAS_BASEMAP, reason="contextily is installed")
def test_missing_basemap_library_degrades_to_a_plain_map_with_a_warning(cfg, tmp_path):
    cfg["output"]["basemap"] = "auto"
    with pytest.warns(UserWarning, match="contextily"):
        render_snapshot(_prediction(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert (tmp_path / "m.png").exists()


@pytest.mark.skipif(HAS_BASEMAP, reason="contextily is installed")
def test_basemap_can_be_made_mandatory(cfg, tmp_path):
    cfg["output"]["basemap"] = "contextily"
    with pytest.raises(ImportError):
        render_snapshot(_prediction(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)


def test_interval_layers_keep_nodata_and_clip_the_lower_bound():
    xy = pd.DataFrame({"location_id": [1], "x": [50.0], "y": [50.0]})
    cal = pd.DataFrame({"location_id": [1] * 40, "score": [5.0] * 40})
    model = GeoCP(0.1, "gaussian", np.inf).fit(cal, xy)
    pred = np.array([[2.0, np.nan], [20.0, 30.0]])
    lower, upper, width = interval_layers(pred, from_origin(0, 20, 10, 10), model, lower_bound=0.0)
    assert lower[0, 0] == 0.0 and upper[0, 0] == 7.0 and width[0, 0] == 7.0              # clipped after forming
    assert np.isnan(lower[0, 1]) and np.isnan(width[0, 1])
    assert width[1, 0] == 10.0 and np.allclose(upper - lower, width, equal_nan=True)


# ------------------------------------------------------------------------------ the black line between "below" and "above"
def _plain_cfg(cfg):
    cfg["output"]["basemap"]["enabled"] = False                       # no tiles, no network: only cells, outline, legend
    return cfg


def _two_zone_prediction():
    """Left half 20 ug/m3 (below 35), right half 60 (above): the 35 line would run down the middle."""
    v = np.full((10, 10), 20.0)
    v[:, 5:] = 60.0
    return v


def _image(path):
    import matplotlib.image as mpimg
    return mpimg.imread(path)


def test_no_outline_is_drawn_by_default(cfg, tmp_path):
    cfg = _plain_cfg(cfg)
    assert cfg["output"]["outline_above_threshold"] is False
    render_snapshot(_two_zone_prediction(), TRANSFORM, "EPSG:32651", str(tmp_path / "with.png"), "t", cfg, 35)
    render_snapshot(_two_zone_prediction(), TRANSFORM, "EPSG:32651", str(tmp_path / "without.png"), "t", cfg, None)
    # a threshold is still passed (it feeds the summary), yet the picture equals one made with no threshold at all
    assert np.array_equal(_image(tmp_path / "with.png"), _image(tmp_path / "without.png"))


def test_the_outline_can_be_switched_back_on(cfg, tmp_path):
    cfg = _plain_cfg(cfg)
    render_snapshot(_two_zone_prediction(), TRANSFORM, "EPSG:32651", str(tmp_path / "off.png"), "t", cfg, 35)
    cfg["output"]["outline_above_threshold"] = True
    render_snapshot(_two_zone_prediction(), TRANSFORM, "EPSG:32651", str(tmp_path / "on.png"), "t", cfg, 35)
    off, on = _image(tmp_path / "off.png"), _image(tmp_path / "on.png")
    assert off.shape == on.shape and (np.abs(off - on).sum(axis=-1) > 0.3).sum() > 100       # a visible dark line plus a legend entry


def test_the_legend_has_no_outline_entry_unless_it_is_on(cfg):
    from output.aqi_maps import _legend_entries
    labels = lambda: [h.get_label() for h in _legend_entries(cfg, 35, "outline: above 35 ug/m3")]
    assert "outline: above 35 ug/m3" not in labels() and len(labels()) == 5                    # only the five AQI bands
    cfg["output"]["outline_above_threshold"] = True
    assert "outline: above 35 ug/m3" in labels()


@pytest.mark.parametrize("flag", [False, True])
def test_panels_and_the_daily_mean_follow_the_same_switch(cfg, tmp_path, flag):
    cfg = _plain_cfg(cfg)
    cfg["output"]["outline_above_threshold"] = flag
    stack = np.stack([_two_zone_prediction()] * 2)
    render_panels(stack, ["08:00", "12:00"], TRANSFORM, "EPSG:32651", str(tmp_path / "p.png"), "day", cfg, 35)
    render_snapshot(_two_zone_prediction(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "mean", cfg, 50)
    render_panels(stack, ["08:00", "12:00"], TRANSFORM, "EPSG:32651", str(tmp_path / "p0.png"), "day", cfg, None)
    render_snapshot(_two_zone_prediction(), TRANSFORM, "EPSG:32651", str(tmp_path / "m0.png"), "mean", cfg, None)
    same = lambda a, b: np.array_equal(_image(tmp_path / a), _image(tmp_path / b))
    assert same("p.png", "p0.png") is (not flag) and same("m.png", "m0.png") is (not flag)


# ------------------------------------------------------------------------------ smooth band edges instead of squares
from output.aqi_maps import display_grid                                  # noqa: E402


def _coarse_cfg(cfg, **kw):
    cfg = _plain_cfg(cfg)
    cfg["output"].update(kw)
    return cfg


def test_smoothing_is_the_default_and_can_be_switched_off(base_cfg, cfg):
    o = base_cfg["output"]
    assert o["smooth_maps"] is True and o["smooth_target_px"] == 1600 and o["smooth_sigma_cells"] == 0.8
    v = np.random.default_rng(0).uniform(10, 60, (20, 30))
    assert np.array_equal(display_grid(v, _coarse_cfg(cfg, smooth_maps=False)), v)            # the raw cells, untouched


def test_a_coarse_grid_is_drawn_on_a_finer_one_over_the_same_area(cfg):
    out = display_grid(np.random.default_rng(0).uniform(10, 60, (20, 30)), _coarse_cfg(cfg))
    k = out.shape[0] // 20
    assert k >= 40 and out.shape == (20 * k, 30 * k)                                           # about 1600 px along the long side


def test_a_grid_that_is_already_fine_is_left_alone(cfg):
    v = np.random.default_rng(0).uniform(10, 60, (1700, 4))
    assert display_grid(v, _coarse_cfg(cfg)) is v or np.array_equal(display_grid(v, _coarse_cfg(cfg)), v)


def test_smoothing_never_invents_values_outside_the_data(cfg):
    v = np.random.default_rng(1).uniform(10, 60, (20, 30))
    out = display_grid(v, _coarse_cfg(cfg, smooth_sigma_cells=0.0))                            # pure spline: the case that could overshoot
    assert out.min() >= v.min() - 1e-9 and out.max() <= v.max() + 1e-9


def test_a_flat_field_stays_flat_and_nan_cells_stay_empty(cfg):
    flat = display_grid(np.full((12, 16), 30.0), _coarse_cfg(cfg))
    assert np.allclose(flat, 30.0)
    v = np.full((20, 30), 30.0)
    v[12:, :8] = np.nan                                                                         # an area without predictions
    out = display_grid(v, _coarse_cfg(cfg))
    k = out.shape[0] // 20
    assert np.isnan(out[14 * k:, : 6 * k]).all()                                                # well inside the empty area
    assert np.isfinite(out[: 10 * k]).all() and np.isfinite(out[:, 12 * k:]).all()              # well inside the predicted area


def test_a_band_edge_stays_where_the_data_puts_it(cfg):
    """Left 15 columns 20 ug/m3 (Good), right 15 columns 40 (unhealthy for sensitive groups): the edge is at column 15."""
    v = np.full((20, 30), 20.0)
    v[:, 15:] = 40.0
    for sigma in (0.0, 0.8):
        out = display_grid(v, _coarse_cfg(cfg, smooth_sigma_cells=sigma))
        k = out.shape[1] // 30
        cat = categorize(out[10], cfg)
        edge = int(np.argmax(cat > 0))                                                          # first display column that is no longer Good
        assert abs(edge - 15 * k) <= 1.5 * k, (sigma, edge / k)                                # within about one source cell of the true edge
        assert np.all(np.diff(cat) >= 0)                                                        # one clean edge, no stripes


def test_the_blur_removes_speckle_from_the_edges(cfg):
    """A noisy field around the 25 ug/m3 edge: the smoothed map crosses it far fewer times along a row."""
    rng = np.random.default_rng(2)
    v = 25 + rng.normal(0, 1.0, (30, 40)) + np.linspace(-6, 6, 40)[None, :]
    crossings = lambda out: int(np.abs(np.diff(categorize(out[out.shape[0] // 2], cfg))).sum())
    raw = crossings(display_grid(v, _coarse_cfg(cfg, smooth_sigma_cells=0.0)))
    soft = crossings(display_grid(v, _coarse_cfg(cfg, smooth_sigma_cells=1.0)))
    assert soft < raw and soft <= 4


def test_the_picture_changes_but_not_its_size_or_area(cfg, tmp_path):
    cfg = _coarse_cfg(cfg)
    v = np.random.default_rng(3).uniform(15, 55, (24, 32))
    render_snapshot(v, TRANSFORM, "EPSG:32651", str(tmp_path / "smooth.png"), "t", cfg, 35)
    cfg["output"]["smooth_maps"] = False
    render_snapshot(v, TRANSFORM, "EPSG:32651", str(tmp_path / "blocky.png"), "t", cfg, 35)
    a, b = _image(tmp_path / "smooth.png"), _image(tmp_path / "blocky.png")
    assert a.shape == b.shape and (np.abs(a - b).sum(axis=-1) > 0.3).sum() > 2000


def test_the_outline_and_the_panels_work_on_the_smoothed_grid(cfg, tmp_path):
    cfg = _coarse_cfg(cfg, outline_above_threshold=True)
    v = np.random.default_rng(4).uniform(15, 55, (24, 32))
    render_snapshot(v, TRANSFORM, "EPSG:32651", str(tmp_path / "o.png"), "t", cfg, 35)
    render_panels(np.stack([v, v]), ["08:00", "12:00"], TRANSFORM, "EPSG:32651", str(tmp_path / "p.png"), "d", cfg, 35)
    assert (tmp_path / "o.png").stat().st_size > 5000 and (tmp_path / "p.png").stat().st_size > 5000
