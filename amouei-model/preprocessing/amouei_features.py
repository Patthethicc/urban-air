"""Inputs taken from the Amouei (2023) framework, each a switch in configs/data.yaml (data.temporal), all OFF by default
so this folder reproduces the main model exactly until one is switched on.

Amouei, M. (2023). Classification of urban morphology and its relationship with air pollution using deep learning.
MSc thesis, ITC, University of Twente. Stage 2 feeds an LSTM a 24-hour window of tabular features per sensor:

  Amouei input                                    switch here                           columns per timestep
  ----------------------------------------------  ------------------------------------  ---------------------------------
  "Time of year" sin/cos                          temporal.cyclical_year: true          sin_doy, cos_doy
  temperature, humidity (nearest met. station)    temporal.weather.enabled: true        temperature_2m, relative_humidity_2m
  wind as a vector (speed + direction -> x, y)    (same, variables wind_u / wind_v)     wind_u, wind_v
  traffic proxy: road density around the sensor   temporal.static_features: [road_density]   road_density
  LCZ label (+ class probabilities) per sensor    temporal.static_features: [lcz_proxy]      lcz_<class> (one-hot)
  morphology as tabular inputs of the LSTM        temporal.static_features: [morphology]     morph_<band> (patch means)

What differs from Amouei, and why:
  - Weather comes from ERA5 reanalysis points (preprocessing/fetch_weather.py), not stations: Manila has no open hourly
    station network for 2023-2026. Each sensor takes its nearest point, as Amouei's "distance to nearest hub".
  - The traffic proxy is the share of street pixels in the sensor's 320 m patch (the street band of the Master
    GeoTIFF, which is > 0 only on streets), close to Amouei's 300 m kernel but without road-class weights.
  - There is no Sentinel-2 LCZ classifier: lcz_proxy assigns a Stewart & Oke (2012) style class from the patch's
    building density, mean height and NDVI with fixed thresholds. It is a stand-in for the LCZ stage, which needs
    labelled training areas this project does not have. It tests the IDEA (discrete urban-form classes as inputs).
  - Static features are repeated at every timestep, so they reach the LSTM exactly like Amouei's tabular columns.
"""
import numpy as np
import pandas as pd

WEATHER_RAW = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation",
               "surface_pressure"]
WEATHER_DERIVED = ["wind_u", "wind_v"]
LCZ_CLASSES = ["compact_high", "compact_mid", "compact_low", "open_high", "open_mid", "open_low",
               "natural_trees", "natural_low_plants", "natural_bare_water"]
STATIC_KINDS = ("morphology", "road_density", "lcz_proxy")


def _t(data_cfg: dict) -> dict:
    return data_cfg["temporal"]


def weather_names(data_cfg: dict) -> list[str]:
    w = _t(data_cfg).get("weather") or {}
    return list(w.get("variables", [])) if w.get("enabled") else []


def static_names(data_cfg: dict) -> list[str]:
    t = _t(data_cfg)
    names = []
    for kind in t.get("static_features") or []:
        if kind == "morphology":
            names += [f"morph_{b}" for b in t.get("morphology_bands", [])]
        elif kind == "road_density":
            names.append("road_density")
        elif kind == "lcz_proxy":
            names += [f"lcz_{c}" for c in LCZ_CLASSES]
        else:
            raise ValueError(f"unknown temporal.static_features entry {kind!r}; use any of {STATIC_KINDS}")
    return names


def year_names(data_cfg: dict) -> list[str]:
    return ["sin_doy", "cos_doy"] if _t(data_cfg).get("cyclical_year") else []


def amouei_names(data_cfg: dict) -> list[str]:
    """The extra per-timestep columns, in the order they follow the main model's features."""
    return year_names(data_cfg) + weather_names(data_cfg) + static_names(data_cfg)


def standardized_names(data_cfg: dict) -> list[str]:
    """Columns z-scored with the training rows' mean / std (sin/cos and one-hot classes are left as they are)."""
    return weather_names(data_cfg) + [n for n in static_names(data_cfg) if not n.startswith("lcz_")]


def any_enabled(data_cfg: dict) -> bool:
    return bool(amouei_names(data_cfg))


# ---------------------------------------------------------------------------------------------------------- time of year
def add_cyclical_year(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """sin/cos(2 pi * day_of_year / 365.25), with the hour as a fraction of the day so the signal is smooth."""
    ts = df[ts_col].dt
    doy = (ts.dayofyear - 1 + ts.hour / 24.0).to_numpy()
    df = df.copy()
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return df


# --------------------------------------------------------------------------------------------------------------- weather
def load_weather(path: str) -> pd.DataFrame:
    w = pd.read_csv(path, parse_dates=["local_timestamp"])
    if {"wind_speed_10m", "wind_direction_10m"} <= set(w.columns):
        # the TensorFlow time-series tutorial Amouei followed: a vector instead of an angle (no jump at 360 -> 0)
        rad = np.deg2rad(w["wind_direction_10m"].to_numpy())
        w["wind_u"] = w["wind_speed_10m"].to_numpy() * np.cos(rad)
        w["wind_v"] = w["wind_speed_10m"].to_numpy() * np.sin(rad)
    return w


def nearest_points(stations: pd.DataFrame, points: pd.DataFrame, crs: str) -> pd.Series:
    """location_id -> point_id of the nearest weather point (planar distance in the projected CRS)."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    px, py = tr.transform(points["longitude"].to_numpy(), points["latitude"].to_numpy())
    _, idx = cKDTree(np.c_[px, py]).query(stations[["x", "y"]].to_numpy())
    return pd.Series(points["point_id"].to_numpy()[idx], index=stations["location_id"].to_numpy())


def add_weather(df: pd.DataFrame, stations: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join the nearest point's hourly weather to every (station, hour). `lead_hours` = 1 shifts the weather one hour
    EARLIER per station, so the step at t-1 carries the weather of hour t (the target hour)."""
    d = cfg["data"]
    wc = d["temporal"]["weather"]
    names = weather_names(d)
    c = d["columns"]
    path = d["paths"]["weather_csv"]
    w = load_weather(path)
    missing = [n for n in names if n not in w.columns]
    if missing:
        raise KeyError(f"weather columns {missing} not in {path} (has {list(w.columns)})")
    points = w.groupby("point_id")[["latitude", "longitude"]].first().reset_index()
    near = nearest_points(stations, points, d["crs"])
    out = df.copy()
    out["_point"] = out[c["group"]].map(near)
    w = w[["point_id", "local_timestamp", *names]].rename(columns={"point_id": "_point", "local_timestamp": c["timestamp"]})
    out = out.merge(w, on=["_point", c["timestamp"]], how="left").drop(columns="_point")
    lead = int(wc.get("lead_hours", 0) or 0)
    if lead:
        out = out.sort_values([c["group"], c["timestamp"]]).reset_index(drop=True)
        out[names] = out.groupby(c["group"])[names].shift(-lead)
    return out


# ------------------------------------------------------------------------------------------------------ static features
def lcz_proxy_class(density: float, height: float, ndvi: float, t: dict | None = None) -> str:
    """A Stewart & Oke (2012) style class from three patch means. Thresholds (configurable under
    temporal.lcz_thresholds): building surface fraction < natural_density -> natural (by NDVI); else open below
    compact_density, compact above; height < low_m low-rise, < high_m mid-rise, else high-rise."""
    t = {"natural_density": 0.10, "compact_density": 0.40, "low_m": 10.0, "high_m": 25.0,
         "trees_ndvi": 0.5, "plants_ndvi": 0.2, **(t or {})}
    if not np.isfinite(density) or density < t["natural_density"]:
        if np.isfinite(ndvi) and ndvi >= t["trees_ndvi"]:
            return "natural_trees"
        if np.isfinite(ndvi) and ndvi >= t["plants_ndvi"]:
            return "natural_low_plants"
        return "natural_bare_water"
    form = "compact" if density >= t["compact_density"] else "open"
    h = height if np.isfinite(height) else 0.0
    rise = "low" if h < t["low_m"] else "mid" if h < t["high_m"] else "high"
    return f"{form}_{rise}"


def _band(bands: list[str], name: str) -> int:
    if name not in bands:
        raise KeyError(f"band {name!r} not in the raster bands {bands}; set temporal.amouei_bands")
    return bands.index(name)


def static_station_features(patches: dict, band_names: list[str], cfg: dict) -> pd.DataFrame:
    """One row per station (location_id + the static columns) from its RAW (unscaled) patch [C, H, W].
    The patch is zero padded at the raster edge and NoData is NaN; both are left out of the means."""
    t = cfg["data"]["temporal"]
    kinds = t.get("static_features") or []
    roles = {"height": "building_height_mean", "density": "building_density", "ndvi": "ndvi_mean",
             "street": "street_aspect_ratio_mean", **(t.get("amouei_bands") or {})}
    rows = []
    for sid, p in patches.items():
        p = np.asarray(p, dtype=np.float64)
        row = {"location_id": int(sid)}
        means = {b: float(np.nanmean(p[i])) if np.isfinite(p[i]).any() else np.nan for i, b in enumerate(band_names)}
        if "morphology" in kinds:
            for b in t.get("morphology_bands", []):
                row[f"morph_{b}"] = means[band_names[_band(band_names, b)]]
        if "road_density" in kinds:
            s = p[_band(band_names, roles["street"])]
            row["road_density"] = float(np.mean(np.nan_to_num(s) > 0))
        if "lcz_proxy" in kinds:
            cls = lcz_proxy_class(means[roles["density"]], means[roles["height"]], means[roles["ndvi"]],
                                  t.get("lcz_thresholds"))
            for c in LCZ_CLASSES:
                row[f"lcz_{c}"] = 1.0 if c == cls else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    names = static_names(cfg["data"])
    bad = out[names].isna().any(axis=1) if names else pd.Series(False, index=out.index)
    if bad.any():                                   # a fully NoData patch: 0 after z-scoring = the training mean
        out.loc[bad, names] = out.loc[bad, names].fillna(out[names].mean())
    return out


def add_static(df: pd.DataFrame, static: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g = cfg["data"]["columns"]["group"]
    s = static.rename(columns={"location_id": g})
    s[g] = s[g].astype(df[g].dtype)
    return df.merge(s, on=g, how="left")


def raster_band_names(path: str, fallback: list[str]) -> list[str]:
    import rasterio
    with rasterio.open(path) as src:
        names = [d for d in src.descriptions]
    return names if all(names) else list(fallback)


# ------------------------------------------------------------------------------------------- weather for raster cells
_WEATHER_CACHE: dict = {}


def cell_weather(cell_xy, cfg: dict) -> dict:
    """{weather column: f(stamp) -> [n_cells] values} for map cells, from the nearest weather point (as for stations).
    `stamp` is the step's hour; lead_hours shifts it forward like add_weather does for stations."""
    d = cfg["data"]
    names = weather_names(d)
    if not names:
        return {}
    path = d["paths"]["weather_csv"]
    if path not in _WEATHER_CACHE:
        w = load_weather(path)
        _WEATHER_CACHE[path] = (w.groupby("point_id")[["latitude", "longitude"]].first().reset_index(),
                                w.set_index(["local_timestamp", "point_id"]).sort_index())
    points, table = _WEATHER_CACHE[path]
    cells = pd.DataFrame({"location_id": np.arange(len(cell_xy)), "x": np.asarray(cell_xy)[:, 0], "y": np.asarray(cell_xy)[:, 1]})
    near = nearest_points(cells, points, d["crs"]).to_numpy()
    lead = pd.Timedelta(hours=int(d["temporal"]["weather"].get("lead_hours", 0) or 0))

    def getter(name):
        def values(stamp):
            try:
                hour = table.loc[stamp + lead, name]
            except KeyError:
                raise ValueError(f"no weather for {stamp + lead} in {path}: run preprocessing.fetch_weather with a later --end")
            return hour.reindex(near).to_numpy(dtype=np.float64)
        return values
    return {n: getter(n) for n in names}
