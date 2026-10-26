import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make model/ importable
os.environ["URBAN_AIR_NO_DOTENV"] = "1"     # tests never read the real model/.env (a real API key must not reach them)

from config_parser import derive, load_config  # noqa: E402


@pytest.fixture(scope="session")
def base_cfg():
    return load_config()


@pytest.fixture
def cfg(base_cfg):
    out = deepcopy(base_cfg)
    out["data"]["paths"]["boundary"] = None        # maps are tested on synthetic coordinates: never clip them to the real NCR
    return out


@pytest.fixture
def small_cfg(cfg):
    """Two lags, no spatial lag, no cyclical hour: easy to check by hand."""
    cfg["data"]["temporal"].update(lags=[2, 1], rolling_window=None, cyclical_hour=False)
    cfg["data"]["spatial_lag"]["enabled"] = False
    return derive(cfg)


@pytest.fixture(autouse=True)
def _reset_basemap_state():
    """The basemap remembers a failed tile attempt for the rest of a run: never let that leak between tests."""
    from output import aqi_maps
    aqi_maps.reset_basemap_state()
    aqi_maps._VECTOR_CACHE.clear()
    yield
    aqi_maps.reset_basemap_state()
