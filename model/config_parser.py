"""Load and validate the YAML configs."""
from copy import deepcopy
from pathlib import Path

import yaml

MODEL_DIR = Path(__file__).parent
CONFIG_DIR = MODEL_DIR / "configs"
SECTIONS = ["data", "model", "train", "report", "geocp", "tune", "inference", "output"]


def load_yaml(name: str) -> dict:
    """Read configs/<name>.yaml (name without extension)."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def feature_base_names(data_cfg: dict) -> list[str]:
    """Per-timestep feature names, in the F-order used everywhere:
    [pm25, (spatial_lag), (roll_mean, roll_std), (sin_hour, cos_hour)]."""
    t = data_cfg["temporal"]
    names = ["pm25"]
    if data_cfg["spatial_lag"]["enabled"]:
        names.append("spatial_lag")
    if t.get("rolling_window"):
        names += ["roll_mean", "roll_std"]
    if t.get("cyclical_hour"):
        names += ["sin_hour", "cos_hour"]
    return names


def _resolve_paths(data_cfg: dict) -> None:
    for key, value in data_cfg["paths"].items():
        if isinstance(value, str):
            p = Path(value)
            data_cfg["paths"][key] = str(p if p.is_absolute() else (MODEL_DIR / p).resolve())


def derive(cfg: dict) -> dict:
    """Fill the values that must not be typed by hand, so configs cannot drift apart:
      T          = len(data.temporal.lags)
      F          = 1 (pm25) + spatial_lag.enabled + 2*(rolling_window set) + 2*cyclical_hour
      C          = band count of data.paths.master_raster (rasterio); falls back to len(raster.band_order)
      patch_px   = round(patch.extent_m / raster resolution)
      model.spatial_cnn.in_channels  = C
      model.spatial_cnn.patch_px     = patch_px (used by feature_map: flatten)
      model.temporal_lstm.input_size = F
    Also records feature_names and the scaler index sets (pm25_feature_idx, rolling_std_idx)."""
    data = cfg["data"]
    names = feature_base_names(data)
    pm25_like = {"pm25", "spatial_lag", "roll_mean", "roll_std"}

    raster_path = Path(data["paths"]["master_raster"] or "")
    resolution = float(data["raster"]["resolution_m"])
    bands = len(data["raster"]["band_order"])
    if raster_path.is_file():
        import rasterio
        with rasterio.open(raster_path) as src:
            bands, resolution = src.count, float(src.res[0])

    derived = {
        "T": len(data["temporal"]["lags"]),
        "F": len(names),
        "C": bands,
        "raster_resolution_m": resolution,
        "patch_px": max(1, int(round(data["patch"]["extent_m"] / resolution))),
        "feature_names": names,
        "pm25_feature_idx": [i for i, n in enumerate(names) if n in pm25_like],
        "rolling_std_idx": [i for i, n in enumerate(names) if n == "roll_std"],
    }
    cfg["derived"] = derived
    cfg["model"]["spatial_cnn"]["in_channels"] = derived["C"]
    cfg["model"]["spatial_cnn"]["patch_px"] = derived["patch_px"]
    cfg["model"]["temporal_lstm"]["input_size"] = derived["F"]
    return cfg


def validate(cfg: dict) -> None:
    from pyproj import CRS

    crs = CRS.from_user_input(cfg["data"]["crs"])
    if not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info):
        raise ValueError(f"data.crs {cfg['data']['crs']} must be a projected CRS in metres (planar distances in m); "
                         "a geographic CRS (degrees, e.g. EPSG:4326) is refused")
    split = cfg["data"]["split"]
    for key in ("test_fraction", "calibration_fraction"):
        if not 0 < split[key] < 1:
            raise ValueError(f"split.{key} must be in (0, 1), got {split[key]}")
    if cfg["model"]["spatial_cnn"]["feature_map"] not in ("pool", "flatten"):
        raise ValueError("spatial_cnn.feature_map must be 'pool' or 'flatten'")
    if cfg["model"]["spatial_cnn"]["kernel_size"] % 2 == 0:
        raise ValueError("spatial_cnn.kernel_size must be odd")
    b = cfg["output"]["basemap"]
    if isinstance(b, dict):
        unknown = set(b) - {"enabled", "web_tiles", "vector_layers", "strict"}
        if unknown or any(not isinstance(v, bool) for v in b.values()):
            raise ValueError("output.basemap takes only true/false for enabled, web_tiles, vector_layers, strict "
                             f"(got {b})")
    elif b not in ("auto", "contextily", "vectors", "none"):
        raise ValueError(f"output.basemap must be a settings block or one of auto | contextily | vectors | none, got {b!r}")
    if not 0 < cfg["geocp"]["alpha"] < 1:
        raise ValueError("geocp.alpha must be in (0, 1)")
    if not cfg["data"]["temporal"]["lags"]:
        raise ValueError("temporal.lags is empty")
    if "recurrent_dropout" in cfg["tune"]["space"] and cfg["model"]["temporal_lstm"]["num_layers"] < 2:
        raise ValueError("tune.space.recurrent_dropout has no effect with temporal_lstm.num_layers = 1")
    for key in cfg["report"]["metrics"] + cfg["report"].get("extra_metrics", []):
        if key not in ("rmse", "mae", "r2", "ia", "pearson_r"):
            raise ValueError(f"unknown metric '{key}' in report.yaml")
    inf = cfg["inference"]
    if inf["stride_px"] < 1 or inf["stride_px"] % 2 == 0:
        raise ValueError("inference.stride_px must be odd (1, 3, 5, ...) so every output cell is centred on a pixel")
    if inf["lag_reference"] not in ("train", "all"):
        raise ValueError("inference.lag_reference must be 'train' or 'all'")
    if cfg["tune"].get("search_epochs"):
        e = cfg["tune"].get("epochs_space") or {}
        if e.get("type") != "integer" or not 1 <= e.get("low", 0) < e.get("high", 0):
            raise ValueError("tune.search_epochs is true: tune.epochs_space needs type integer and 1 <= low < high")
    if cfg["tune"]["n_initial_points"] > cfg["tune"]["n_calls"]:
        raise ValueError("tune.n_initial_points must be <= tune.n_calls")


def validate_for_tuning(cfg: dict) -> None:
    """Checks that only matter for `--stage tune`."""
    if not cfg["model"]["spatial_cnn"].get("enabled", True):
        cnn = [k for k in cfg["tune"]["space"] if k.startswith("cnn_")]
        if cnn:
            raise ValueError(f"spatial_cnn.enabled is false but tune.space searches CNN parameters {cnn}")


def resolve_path(value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (MODEL_DIR / p).resolve())


def _set_overrides(cfg: dict, overrides: list[str]) -> None:
    for item in overrides or []:
        path, _, raw = item.partition("=")
        keys = path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        if keys[-1] not in node:
            raise KeyError(f"unknown config key: {path}")
        node[keys[-1]] = yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """`section.key.sub=value` strings (value parsed as YAML), e.g. train.epochs=2. Re-derives afterwards."""
    _set_overrides(cfg, overrides)
    return derive(cfg)


def load_config(overrides: list[str] | None = None) -> dict:
    """Return {'data','model','train','report','geocp','derived',...} with absolute paths and derived values.
    `.env` (API keys) is loaded into the environment first; --override values are applied before paths resolve."""
    from env_loader import load_dotenv
    load_dotenv()                                  # model/.env -> environment (API keys); never written into the config
    cfg = {name: load_yaml(name) for name in SECTIONS}
    _set_overrides(cfg, overrides or [])
    _resolve_paths(cfg["data"])
    derive(cfg)
    validate(cfg)
    cfg["overrides"] = list(overrides or [])
    return cfg


def check_raster_crs(cfg: dict) -> None:
    """The Master GeoTIFF must be in data.crs: patches, sensor positions and every GeoCP distance (metres)
    assume one projected CRS from preprocessing to inference."""
    import rasterio
    from rasterio.crs import CRS

    with rasterio.open(cfg["data"]["paths"]["master_raster"]) as src:
        if src.crs is None or CRS.from_user_input(cfg["data"]["crs"]) != src.crs:
            raise ValueError(f"Master GeoTIFF is {src.crs} but data.crs is {cfg['data']['crs']}: "
                             "reproject the raster or change data.crs (one CRS for the whole pipeline)")


def snapshot(cfg: dict) -> dict:
    """Plain, JSON-serialisable copy of the whole config (for run_info.json)."""
    return deepcopy(cfg)
