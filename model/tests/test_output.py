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
