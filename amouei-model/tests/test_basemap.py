import importlib.util
import sys
import types
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon

from output import aqi_maps
from output.aqi_maps import render_snapshot

TRANSFORM = from_origin(0, 100, 10, 10)
HAS_BASEMAP = importlib.util.find_spec("contextily") is not None


@pytest.fixture(autouse=True)
def _fresh_basemap_state():
    aqi_maps.reset_basemap_state()
    aqi_maps._VECTOR_CACHE.clear()
    yield
    aqi_maps.reset_basemap_state()


def _pred():
    return np.linspace(10, 60, 100).reshape(10, 10)


def _vectors(tmp_path, crs):
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 50), (100, 50)]), LineString([(50, 0), (50, 100)])], crs=crs)
    boundary = gpd.GeoDataFrame(geometry=[Polygon([(5, 5), (95, 5), (95, 95), (5, 95)])], crs=crs)
    streets.to_file(tmp_path / "streets.geojson", driver="GeoJSON")
    boundary.to_file(tmp_path / "boundary.geojson", driver="GeoJSON")
    return str(tmp_path / "streets.geojson"), str(tmp_path / "boundary.geojson")


def test_vector_basemap_draws_streets_and_boundary_offline(cfg, tmp_path):
    cfg["data"]["crs"] = "EPSG:32651"
    cfg["data"]["paths"]["streets"], cfg["data"]["paths"]["boundary"] = _vectors(tmp_path, "EPSG:32651")
    cfg["output"]["basemap"] = "vectors"
    aqi_maps._VECTOR_CACHE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert (tmp_path / "m.png").stat().st_size > 5000


def test_vector_basemap_is_reprojected_to_the_map_crs(cfg, tmp_path):
    """Layers stored in WGS84 land in the right place of a projected map (no warning = something was drawn)."""
    from pyproj import Transformer
    t = Transformer.from_crs(32651, 4326, always_xy=True)
    lon, lat = t.transform([0, 100], [50, 50])
    line = gpd.GeoDataFrame(geometry=[LineString(list(zip(lon, lat)))], crs="EPSG:4326")
    line.to_file(tmp_path / "streets.geojson", driver="GeoJSON")
    cfg["data"]["crs"] = "EPSG:32651"
    cfg["data"]["paths"].update(streets=str(tmp_path / "streets.geojson"), boundary=None)
    cfg["output"]["basemap"] = "vectors"
    aqi_maps._VECTOR_CACHE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)


def test_vector_basemap_without_layers_warns_and_still_renders(cfg, tmp_path):
    cfg["data"]["paths"].update(streets=None, boundary=None)
    cfg["output"]["basemap"] = "vectors"
    with pytest.warns(UserWarning, match="streets"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert (tmp_path / "m.png").exists()


def test_web_tiles_are_requested_in_the_map_crs_with_attribution_on(cfg, tmp_path, monkeypatch):
    """contextily is faked here (no network): checks HOW it is called - map CRS, provider, attribution not disabled."""
    calls = {}
    fake = types.ModuleType("contextily")
    fake.providers = {"CartoDB": {"Positron": "PROVIDER"}}
    fake.add_basemap = lambda ax, **kw: calls.update(kw)
    monkeypatch.setitem(sys.modules, "contextily", fake)
    cfg["output"]["basemap"] = "contextily"
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert calls["crs"] == "EPSG:32651" and calls["source"] == "PROVIDER" and calls.get("attribution", True) is not False


def test_offline_tile_failure_degrades_to_a_plain_map(cfg, tmp_path, monkeypatch):
    fake = types.ModuleType("contextily")
    fake.providers = {"CartoDB": {"Positron": "PROVIDER"}}
    def boom(ax, **kw):
        raise ConnectionError("no internet")
    fake.add_basemap = boom
    monkeypatch.setitem(sys.modules, "contextily", fake)
    cfg["output"]["basemap"] = "auto"
    with pytest.warns(UserWarning, match="basemap could not be loaded"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert (tmp_path / "m.png").exists()


def _fake_contextily(monkeypatch, add_basemap, providers=None, set_cache_dir=None):
    fake = types.ModuleType("contextily")
    fake.providers = providers or {"CartoDB": {"Positron": "PROVIDER"}}
    fake.add_basemap = add_basemap
    if set_cache_dir:
        fake.set_cache_dir = set_cache_dir
    monkeypatch.setitem(sys.modules, "contextily", fake)


def _vector_cfg(cfg, tmp_path, mode):
    cfg["data"]["crs"] = "EPSG:32651"
    cfg["data"]["paths"]["streets"], cfg["data"]["paths"]["boundary"] = _vectors(tmp_path, "EPSG:32651")
    cfg["output"]["basemap"] = mode


def test_auto_prefers_web_tiles_when_they_work(cfg, tmp_path, monkeypatch):
    calls = []
    _fake_contextily(monkeypatch, lambda ax, **kw: calls.append(kw))
    _vector_cfg(cfg, tmp_path, "auto")
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert len(calls) == 1 and aqi_maps.last_basemap() == "web"


def test_auto_falls_back_to_the_offline_vectors_when_tiles_fail(cfg, tmp_path, monkeypatch):
    def boom(ax, **kw):
        raise ConnectionError("no internet")
    _fake_contextily(monkeypatch, boom)
    _vector_cfg(cfg, tmp_path, "auto")
    with pytest.warns(UserWarning, match="could not be loaded"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert aqi_maps.last_basemap() == "vectors" and (tmp_path / "m.png").stat().st_size > 5000


def test_offline_machine_is_detected_before_any_tile_request(cfg, tmp_path, monkeypatch):
    calls = []
    provider = {"url": "https://{s}.basemaps.example.com/{z}/{x}/{y}.png"}
    _fake_contextily(monkeypatch, lambda ax, **kw: calls.append(kw), providers={"CartoDB": {"Positron": provider}})
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    _vector_cfg(cfg, tmp_path, "auto")
    with pytest.warns(UserWarning, match="not reachable"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert calls == [] and aqi_maps.last_basemap() == "vectors"


def test_after_one_tile_failure_the_rest_of_the_run_skips_the_web(cfg, tmp_path, monkeypatch):
    calls = []

    def boom(ax, **kw):
        calls.append(1)
        raise ConnectionError("no internet")
    _fake_contextily(monkeypatch, boom)
    _vector_cfg(cfg, tmp_path, "auto")
    with pytest.warns(UserWarning):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "a.png"), "t", cfg, 35)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                              # no second warning, no second attempt
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "b.png"), "t", cfg, 35)
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "c.png"), "t", cfg, 35)
    assert len(calls) == 1 and aqi_maps.last_basemap() == "vectors"


@pytest.mark.skipif(HAS_BASEMAP, reason="contextily is installed")
def test_auto_without_the_library_uses_the_vectors_quietly_after_one_notice(cfg, tmp_path):
    _vector_cfg(cfg, tmp_path, "auto")
    with pytest.warns(UserWarning, match="contextily"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "a.png"), "t", cfg, 35)
    assert aqi_maps.last_basemap() == "vectors"


@pytest.mark.skipif(HAS_BASEMAP, reason="contextily is installed")
def test_auto_with_nothing_available_still_renders_a_plain_map(cfg, tmp_path):
    cfg["data"]["paths"].update(streets=None, boundary=None)
    cfg["output"]["basemap"] = "auto"
    with pytest.warns(UserWarning, match="contextily"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert aqi_maps.last_basemap() is None and (tmp_path / "m.png").exists()


def test_an_unreadable_vector_file_does_not_stop_the_run(cfg, tmp_path):
    cfg["data"]["paths"].update(streets=str(tmp_path / "missing.geojson"), boundary=None)
    cfg["output"]["basemap"] = "auto"
    cfg["output"]["basemap_provider"] = "CartoDB.Positron"
    import importlib.util as u
    if u.find_spec("contextily") is None:
        with pytest.warns(UserWarning, match="contextily"):
            render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert aqi_maps.last_basemap() is None


def test_tiles_are_cached_on_disk_for_later_offline_use(cfg, tmp_path, monkeypatch):
    seen = []
    _fake_contextily(monkeypatch, lambda ax, **kw: None, set_cache_dir=lambda path: seen.append(path))
    cfg["output"]["basemap"] = "contextily"
    cfg["output"]["basemap_cache_dir"] = str(tmp_path / "tiles")
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)
    assert seen == [str(tmp_path / "tiles")]


def test_strict_contextily_mode_still_demands_the_library(cfg, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "contextily", None)            # makes `import contextily` raise ImportError
    cfg["output"]["basemap"] = "contextily"
    with pytest.raises(ImportError):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, 35)


def _pixels(path):
    import matplotlib.image as mpimg
    return mpimg.imread(path)


def test_vector_basemap_is_actually_visible_on_the_map(cfg, tmp_path):
    """Regression: the layers were once drawn under the opaque cells and the map looked identical. Compare pixels."""
    sensors = pd.DataFrame({"x": [], "y": []})
    cfg["data"]["crs"] = "EPSG:32651"
    cfg["data"]["paths"]["streets"], cfg["data"]["paths"]["boundary"] = _vectors(tmp_path, "EPSG:32651")
    cfg["output"]["basemap"] = "none"
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "plain.png"), "t", cfg, None, sensors)
    cfg["output"]["basemap"] = "vectors"
    aqi_maps._VECTOR_CACHE.clear()
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "vec.png"), "t", cfg, None, sensors)
    a, b = _pixels(tmp_path / "plain.png"), _pixels(tmp_path / "vec.png")
    assert a.shape == b.shape and (np.abs(a - b).sum(axis=-1) > 0.2).sum() > 300       # hundreds of pixels differ


def test_auto_fallback_is_visible_too(cfg, tmp_path):
    cfg["data"]["crs"] = "EPSG:32651"
    cfg["data"]["paths"]["streets"], cfg["data"]["paths"]["boundary"] = _vectors(tmp_path, "EPSG:32651")
    cfg["output"]["basemap"] = "none"
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "plain.png"), "t", cfg, None)
    cfg["output"]["basemap"] = "auto"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "auto.png"), "t", cfg, None)
    if not HAS_BASEMAP:
        assert aqi_maps.last_basemap() == "vectors"
        assert (np.abs(_pixels(tmp_path / "plain.png") - _pixels(tmp_path / "auto.png")).sum(axis=-1) > 0.2).sum() > 300


# ----------------------------------------------------------------------------- independent switches
def _switch(cfg, **kw):
    cfg["output"]["basemap"] = {"enabled": True, "web_tiles": True, "vector_layers": True, "strict": False, **kw}


def _no_network(monkeypatch):
    def guard(*a, **k):
        raise AssertionError("the network must not be touched")
    monkeypatch.setattr("socket.create_connection", guard)


def test_default_settings_are_the_full_fallback_chain(base_cfg):
    assert base_cfg["output"]["basemap"] == {"enabled": True, "web_tiles": True, "vector_layers": True, "strict": False}
    assert aqi_maps.basemap_settings(base_cfg) == base_cfg["output"]["basemap"]


def test_legacy_mode_strings_still_map_onto_the_switches(cfg):
    expected = {"auto": (True, True, True, False), "contextily": (True, True, False, True),
                "vectors": (True, False, True, False), "none": (False, False, False, False)}
    for mode, flags in expected.items():
        cfg["output"]["basemap"] = mode
        got = aqi_maps.basemap_settings(cfg)
        assert (got["enabled"], got["web_tiles"], got["vector_layers"], got["strict"]) == flags, mode


def test_master_switch_off_gives_a_plain_map_and_never_looks_at_tiles_or_layers(cfg, tmp_path, monkeypatch):
    calls = []
    _fake_contextily(monkeypatch, lambda ax, **kw: calls.append(kw))
    _no_network(monkeypatch)
    _vector_cfg(cfg, tmp_path, "auto")
    _switch(cfg, enabled=False)
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "off.png"), "t", cfg, None)
    cfg["output"]["basemap"] = "none"
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "none.png"), "t", cfg, None)
    assert calls == [] and aqi_maps.last_basemap() is None
    assert np.array_equal(_pixels(tmp_path / "off.png"), _pixels(tmp_path / "none.png"))


def test_web_tiles_off_never_touches_the_network_and_uses_the_offline_layers(cfg, tmp_path, monkeypatch):
    calls = []
    _fake_contextily(monkeypatch, lambda ax, **kw: calls.append(kw))
    _no_network(monkeypatch)
    _vector_cfg(cfg, tmp_path, "auto")
    _switch(cfg, web_tiles=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert calls == [] and aqi_maps.last_basemap() == "vectors"


def test_vector_layers_off_means_tiles_or_plain_even_when_layers_are_configured(cfg, tmp_path, monkeypatch):
    def boom(ax, **kw):
        raise ConnectionError("no internet")
    _fake_contextily(monkeypatch, boom)
    _vector_cfg(cfg, tmp_path, "auto")
    _switch(cfg, vector_layers=False)
    with pytest.warns(UserWarning, match="could not be loaded"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert aqi_maps.last_basemap() is None                         # not 'vectors': that source is switched off


def test_vector_layers_off_still_allows_tiles(cfg, tmp_path, monkeypatch):
    calls = []
    _fake_contextily(monkeypatch, lambda ax, **kw: calls.append(kw))
    _switch(cfg, vector_layers=False)
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert len(calls) == 1 and aqi_maps.last_basemap() == "web"


def test_enabled_with_both_sources_off_warns_and_draws_plain(cfg, tmp_path):
    _switch(cfg, web_tiles=False, vector_layers=False)
    with pytest.warns(UserWarning, match="both off"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert aqi_maps.last_basemap() is None


def test_strict_stops_the_run_when_tiles_are_unreachable(cfg, tmp_path, monkeypatch):
    provider = {"url": "https://{s}.basemaps.example.com/{z}/{x}/{y}.png"}
    _fake_contextily(monkeypatch, lambda ax, **kw: None, providers={"CartoDB": {"Positron": provider}})
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    _vector_cfg(cfg, tmp_path, "auto")
    _switch(cfg, strict=True)                                      # vectors would be available, but strict does not fall back
    with pytest.raises(ConnectionError, match="not reachable"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)


def test_strict_stops_the_run_when_a_tile_request_fails(cfg, tmp_path, monkeypatch):
    def boom(ax, **kw):
        raise RuntimeError("tile error")
    _fake_contextily(monkeypatch, boom)
    _switch(cfg, strict=True)
    with pytest.raises(RuntimeError, match="tile error"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)


def test_strict_stops_the_run_when_the_vector_layers_are_missing(cfg, tmp_path):
    cfg["data"]["paths"].update(streets=None, boundary=None)
    _switch(cfg, web_tiles=False, strict=True)
    with pytest.raises(ValueError, match="strict"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)


def test_strict_with_working_sources_just_works(cfg, tmp_path):
    _vector_cfg(cfg, tmp_path, "auto")
    _switch(cfg, web_tiles=False, strict=True)
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert aqi_maps.last_basemap() == "vectors"


def test_the_switches_can_be_flipped_per_run_from_the_command_line():
    from config_parser import load_config
    cfg = load_config(["output.basemap.web_tiles=false", "output.basemap.strict=true"])
    assert aqi_maps.basemap_settings(cfg) == {"enabled": True, "web_tiles": False, "vector_layers": True, "strict": True}


def test_a_bad_basemap_setting_is_rejected(cfg):
    from config_parser import validate
    cfg["output"]["basemap"] = {"enabled": True, "web_tiles": "yes"}
    with pytest.raises(ValueError, match="basemap"):
        validate(cfg)
    cfg["output"]["basemap"] = {"enabled": True, "tiles": True}
    with pytest.raises(ValueError, match="basemap"):
        validate(cfg)
    cfg["output"]["basemap"] = "satellite"
    with pytest.raises(ValueError, match="basemap"):
        validate(cfg)
