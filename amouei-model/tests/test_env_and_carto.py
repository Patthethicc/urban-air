import json
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from config_parser import load_config
from env_loader import ENV_FILE, load_dotenv, parse_env
from output import aqi_maps
from output.aqi_maps import render_snapshot

MODEL_DIR = Path(__file__).resolve().parents[1]
TRANSFORM = from_origin(0, 100, 10, 10)
SECRET = "k3y-SECRET-123"
CARTO = {"url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", "attribution": "(C) OSM (C) CARTO"}
OSM = {"url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "attribution": "(C) OSM"}


def _pred():
    return np.linspace(10, 60, 100).reshape(10, 10)


# ------------------------------------------------------------------------------------------------- .env handling
def test_parser_handles_comments_quotes_export_and_empty_values():
    text = """
# a comment
CARTO_API_KEY = abc123            # trailing comment
export OTHER="with spaces"
SINGLE='quoted'
EMPTY=
no_equals_sign
=novalue
"""
    assert parse_env(text) == {"CARTO_API_KEY": "abc123", "OTHER": "with spaces", "SINGLE": "quoted"}


def test_loading_never_overrides_the_real_environment(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("A_KEY=from_file\nB_KEY=also_from_file\n")
    monkeypatch.setenv("A_KEY", "from_shell")
    monkeypatch.delenv("B_KEY", raising=False)
    applied = load_dotenv(f)
    import os
    assert os.environ["A_KEY"] == "from_shell" and os.environ["B_KEY"] == "also_from_file"
    assert applied == {"B_KEY": "also_from_file"}
    monkeypatch.delenv("B_KEY", raising=False)


def test_a_missing_or_empty_dotenv_is_fine(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}
    (tmp_path / "e.env").write_text("CARTO_API_KEY=\n")
    assert load_dotenv(tmp_path / "e.env") == {}


def test_the_committed_template_holds_no_key_and_the_real_dotenv_is_never_tracked():
    """.env.example is committed, so it must stay empty. .env is YOUR file (it may hold a real key): it only has
    to exist for convenience and must never be tracked by git."""
    assert parse_env((MODEL_DIR / ".env.example").read_text(encoding="utf-8")) == {}
    try:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", ".env"], cwd=MODEL_DIR, capture_output=True,
                                 stdin=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        pytest.skip("git not available")
    assert not tracked


def test_dotenv_is_git_ignored_but_the_template_is_not():
    ignore = (MODEL_DIR / ".gitignore").read_text().splitlines()
    assert ".env" in ignore and ".env.example" not in ignore
    try:
        run = lambda p: subprocess.run(["git", "check-ignore", "-q", p], cwd=MODEL_DIR, capture_output=True,
                                       stdin=subprocess.DEVNULL).returncode
        assert run(".env") == 0 and run(".env.example") == 1
    except FileNotFoundError:
        pytest.skip("git not available")


def test_the_key_never_lands_in_the_loaded_config(monkeypatch):
    monkeypatch.setenv("CARTO_API_KEY", SECRET)
    assert SECRET not in json.dumps(load_config(), default=str)


# ------------------------------------------------------------------------------------------------- key handling
@pytest.fixture(autouse=True)
def _no_network_and_clean_env(monkeypatch):
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: types.SimpleNamespace(close=lambda: None))
    for name in ("CARTO_API_KEY", "MY_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _fake_contextily(monkeypatch, provider, add_basemap):
    fake = types.ModuleType("contextily")
    fake.providers = {"CartoDB": {"Positron": provider}, "OpenStreetMap": {"Mapnik": OSM}}
    fake.add_basemap = add_basemap
    monkeypatch.setitem(sys.modules, "contextily", fake)


def _tiles(cfg, **extra):
    cfg["output"]["basemap"] = {"enabled": True, "web_tiles": True, "vector_layers": False, "strict": False}
    cfg["output"].update(extra)


def test_a_carto_provider_gets_the_key_from_the_environment(cfg, tmp_path, monkeypatch):
    calls = []
    _fake_contextily(monkeypatch, CARTO, lambda ax, **kw: calls.append(kw))
    monkeypatch.setenv("CARTO_API_KEY", SECRET)
    _tiles(cfg)
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert len(calls) == 1 and calls[0]["source"]["url"].endswith(f".png?key={SECRET}")
    assert calls[0]["source"]["attribution"] == CARTO["attribution"]               # attribution kept
    assert aqi_maps.last_basemap() == "web"


def test_the_query_parameter_name_is_configurable(cfg, tmp_path, monkeypatch):
    calls = []
    _fake_contextily(monkeypatch, CARTO, lambda ax, **kw: calls.append(kw))
    monkeypatch.setenv("CARTO_API_KEY", SECRET)
    _tiles(cfg, carto_key_param="api_key")
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert calls[0]["source"]["url"].endswith(f".png?api_key={SECRET}")


def test_without_a_key_carto_tiles_are_never_requested(cfg, tmp_path, monkeypatch):
    calls = []
    _fake_contextily(monkeypatch, CARTO, lambda ax, **kw: calls.append(kw))
    _tiles(cfg)
    with pytest.warns(UserWarning, match="CARTO_API_KEY"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert calls == [] and aqi_maps.last_basemap() is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")                              # the rest of the run stays quiet
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "n.png"), "t", cfg, None)
    assert calls == []


def test_without_a_key_strict_mode_stops_the_run(cfg, tmp_path, monkeypatch):
    _fake_contextily(monkeypatch, CARTO, lambda ax, **kw: None)
    _tiles(cfg)
    cfg["output"]["basemap"]["strict"] = True
    with pytest.raises(RuntimeError, match="CARTO_API_KEY"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)


def test_without_a_key_the_chain_falls_through_to_the_offline_layers(cfg, tmp_path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import LineString
    gpd.GeoDataFrame(geometry=[LineString([(0, 50), (100, 50)])], crs="EPSG:32651").to_file(tmp_path / "s.geojson", driver="GeoJSON")
    _fake_contextily(monkeypatch, CARTO, lambda ax, **kw: None)
    cfg["data"]["crs"] = "EPSG:32651"
    cfg["data"]["paths"].update(streets=str(tmp_path / "s.geojson"), boundary=None)
    cfg["output"]["basemap"] = {"enabled": True, "web_tiles": True, "vector_layers": True, "strict": False}
    with pytest.warns(UserWarning, match="CARTO_API_KEY"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert aqi_maps.last_basemap() == "vectors"


def test_a_provider_that_needs_no_key_works_without_one(cfg, tmp_path, monkeypatch):
    calls = []
    _fake_contextily(monkeypatch, CARTO, lambda ax, **kw: calls.append(kw))
    _tiles(cfg, basemap_provider="OpenStreetMap.Mapnik")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert calls[0]["source"]["url"] == OSM["url"]                  # untouched, no ?key=


def test_the_key_is_removed_from_a_failure_warning(cfg, tmp_path, monkeypatch):
    def boom(ax, **kw):
        raise RuntimeError(f"HTTP 403 for {kw['source']['url']}")
    _fake_contextily(monkeypatch, CARTO, boom)
    monkeypatch.setenv("CARTO_API_KEY", SECRET)
    _tiles(cfg)
    with pytest.warns(UserWarning, match="could not be loaded") as record:
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    text = " ".join(str(w.message) for w in record)
    assert SECRET not in text and "key=***" in text


def test_the_key_is_removed_from_a_strict_error_and_its_traceback_chain(cfg, tmp_path, monkeypatch):
    def boom(ax, **kw):
        raise RuntimeError(f"HTTP 403 for {kw['source']['url']}")
    _fake_contextily(monkeypatch, CARTO, boom)
    monkeypatch.setenv("CARTO_API_KEY", SECRET)
    _tiles(cfg)
    cfg["output"]["basemap"]["strict"] = True
    with pytest.raises(RuntimeError) as err:
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert SECRET not in str(err.value) and "key=***" in str(err.value) and err.value.__cause__ is None


def test_an_explicit_url_template_is_filled_from_the_environment(cfg, tmp_path, monkeypatch):
    calls = []
    _fake_contextily(monkeypatch, CARTO, lambda ax, **kw: calls.append(kw))
    monkeypatch.setenv("MY_TOKEN", SECRET)
    _tiles(cfg, basemap_url="https://tiles.example.com/{z}/{x}/{y}.png?token=${MY_TOKEN}", basemap_attribution="(C) me")
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert calls[0]["source"] == f"https://tiles.example.com/{{z}}/{{x}}/{{y}}.png?token={SECRET}"
    assert calls[0]["attribution"] == "(C) me"


def test_an_explicit_url_template_with_an_unset_variable_is_not_requested(cfg, tmp_path, monkeypatch):
    calls = []
    _fake_contextily(monkeypatch, CARTO, lambda ax, **kw: calls.append(kw))
    _tiles(cfg, basemap_url="https://tiles.example.com/{z}/{x}/{y}.png?token=${MY_TOKEN}")
    with pytest.warns(UserWarning, match="MY_TOKEN"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert calls == []


def test_the_summary_records_the_switches_but_never_the_key(cfg, monkeypatch):
    monkeypatch.setenv("CARTO_API_KEY", SECRET)
    assert SECRET not in json.dumps(aqi_maps.basemap_settings(cfg))


# ------------------------------------------------------------------------------------------------- real libraries, offline
def test_the_test_suite_never_reads_the_real_dotenv(monkeypatch):
    import os
    assert os.environ.get("URBAN_AIR_NO_DOTENV") == "1"
    monkeypatch.delenv("CARTO_API_KEY", raising=False)
    assert load_dotenv() == {}                                      # no path given: skipped under the flag


def test_the_real_xyzservices_carto_provider_is_recognised_and_keyed():
    xyz = pytest.importorskip("xyzservices")
    provider = xyz.providers.CartoDB.Positron
    assert aqi_maps._is_carto(provider) and not aqi_maps._is_carto(xyz.providers.OpenStreetMap.Mapnik)
    if "{apikey}" not in provider.url:
        pytest.skip("this xyzservices ships CartoDB.Positron without a key placeholder; the next test covers that path")
    assert "?key=" in provider.url                                     # xyzservices confirms the parameter name: key
    keyed = aqi_maps._with_key(provider, SECRET, "key")
    url = keyed.build_url(x=1, y=2, z=3)
    assert url.count("?key=") == 1 and url.endswith(f"?key={SECRET}") and "/3/1/2" in url and "{" not in url
    assert "light_all" in url and keyed.attribution == provider.attribution
    assert SECRET not in provider.url and provider.get("apikey") != SECRET            # the original provider is untouched


def test_a_carto_template_without_a_key_placeholder_gets_the_key_appended():
    """An older xyzservices (Kaggle's) ships CartoDB.Positron without `?key={apikey}`: the key is still sent, and only on the copy."""
    xyz = pytest.importorskip("xyzservices")
    old = xyz.TileProvider(name="CartoDB.Positron", url="https://{s}.basemaps.cartocdn.com/{variant}/{z}/{x}/{y}{r}.png",
                           attribution="x", variant="light_all", subdomains="abcd", max_zoom=20)
    assert "{apikey}" not in old.url and aqi_maps._is_carto(old)
    keyed = aqi_maps._with_key(old, SECRET, "key")
    assert keyed.url.endswith(f".png?key={SECRET}") and SECRET not in old.url


def test_real_contextily_receives_the_keyed_provider(cfg, tmp_path, monkeypatch):
    cx = pytest.importorskip("contextily")
    calls = []
    monkeypatch.setattr(cx, "add_basemap", lambda ax, **kw: calls.append(kw))     # no tiles are fetched
    monkeypatch.setenv("CARTO_API_KEY", SECRET)
    _tiles(cfg, basemap_cache_dir=str(tmp_path / "tiles"))
    render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert len(calls) == 1 and calls[0]["source"].build_url(x=1, y=2, z=3).endswith(f"?key={SECRET}")
    assert aqi_maps.last_basemap() == "web"


def test_real_contextily_without_a_key_makes_no_request(cfg, tmp_path, monkeypatch):
    cx = pytest.importorskip("contextily")
    calls = []
    monkeypatch.setattr(cx, "add_basemap", lambda ax, **kw: calls.append(kw))
    _tiles(cfg, basemap_cache_dir=str(tmp_path / "tiles"))
    with pytest.warns(UserWarning, match="CARTO_API_KEY"):
        render_snapshot(_pred(), TRANSFORM, "EPSG:32651", str(tmp_path / "m.png"), "t", cfg, None)
    assert calls == []


def test_affine_composition_works_on_affine_2_and_3():
    from affine import Affine
    from affine_compat import compose
    t = compose(Affine.translation(3, 4), Affine.scale(2))
    assert tuple(t)[:6] == (2.0, 0.0, 3.0, 0.0, 2.0, 4.0)
    assert compose(Affine.translation(1, 1)) == Affine.translation(1, 1)
    base = Affine(10, 0, 100, 0, -10, 500)
    assert compose(base, Affine.translation(2, 3), Affine.scale(5)) * (0, 0) == pytest.approx(base * (2, 3))
