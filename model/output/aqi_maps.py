"""Prediction surfaces as maps on the Philippine AQI scale (Output.txt).

Cells above the highlight threshold (35 ug/m3) are outlined only when output.outline_above_threshold is true (off by default). A basemap of the NCR is drawn when
`contextily` is installed (output.yaml basemap: auto); otherwise the map keeps a plain background.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from output import cartography  # noqa: E402
from reporting.plotters import INK, INK2, SURFACE  # noqa: E402


def categorize(values: np.ndarray, cfg: dict) -> np.ndarray:
    """AQI category index per cell (0 = Good ...), NaN where there is no prediction. A value equal to an
    upper bound belongs to the lower category (Fair is 25.1-35.0)."""
    bins = cfg["output"]["aqi"]["bins"]
    uppers = [b["upper"] for b in bins[:-1]]
    idx = np.digitize(values, uppers, right=True).astype(float)
    idx[~np.isfinite(values)] = np.nan
    return idx


def extent_of(transform, shape) -> tuple:
    left, top = transform.c, transform.f
    return left, left + transform.a * shape[1], top + transform.e * shape[0], top


def display_grid(values: np.ndarray, cfg: dict) -> np.ndarray:
    """The grid actually drawn. With output.smooth_maps (default true) a coarse grid is interpolated onto a finer one
    (bicubic spline, within the range of the data) BEFORE the AQI colours are assigned, so the edges between colour
    bands run as smooth curves instead of staircases of squares. Cells without a prediction stay empty. This is display
    only: the GeoTIFFs and the numbers in output_summary.json keep the raw per-cell values.

    A grid that already has about `smooth_target_px` cells along its longest side (or more) is drawn as it is."""
    o = cfg["output"]
    values = np.asarray(values, dtype=np.float64)
    if not o.get("smooth_maps", True) or values.ndim != 2:
        return values
    valid = np.isfinite(values)
    if valid.sum() < 4:
        return values
    target = int(o.get("smooth_target_px", 1600))
    cap = int(np.sqrt(float(o.get("smooth_max_cells", 6_000_000)) / values.size))            # keep the drawn grid affordable
    k = min(int(np.ceil(target / max(values.shape))), cap)
    if k <= 1:
        return values
    lo, hi = float(values[valid].min()), float(values[valid].max())
    if hi > lo:
        fill = values
        if not valid.all():                                          # empty cells take their nearest value for the spline only
            nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
            fill = values[tuple(nearest)]
        sigma = float(o.get("smooth_sigma_cells", 0.0))
        if sigma > 0:                                                # optional: soften cell-to-cell speckle before interpolating
            fill = ndimage.gaussian_filter(fill, sigma, mode="nearest")
        up = np.clip(ndimage.zoom(fill, k, order=3, mode="nearest", grid_mode=True), lo, hi)
    else:
        up = np.full((values.shape[0] * k, values.shape[1] * k), lo)
    if not valid.all():
        keep = ndimage.zoom(valid.astype(np.float64), k, order=1, mode="nearest", grid_mode=True) >= 0.5
        up = np.where(keep, up, np.nan)
    return up


def _outline_on(cfg: dict) -> bool:
    """The black line between 'below' and 'above' the threshold is an option, off by default (always on in the XGBoost look, whose
    AQI map outlines the cells above 35)."""
    return bool(cfg["output"].get("outline_above_threshold", False)) or xgboost_look(cfg)


def _legend_entries(cfg: dict, threshold, threshold_label: str):
    bins = cfg["output"]["aqi"]["bins"]
    lo, handles = 0, []
    for b in bins:
        rng = f">{lo:g}" if b["upper"] is None else f"{lo:g}-{b['upper']:g}"
        handles.append(Patch(facecolor=b["color"], edgecolor=INK2, linewidth=0.5, label=f"{b['label']} ({rng})"))
        lo = b["upper"]
    if threshold is not None and _outline_on(cfg):
        handles.append(Line2D([0], [0], color=INK, linewidth=1.2, label=threshold_label))
    if _STATE["last"] == "vectors":                              # name the offline basemap layers that were drawn
        paths = cfg["data"]["paths"]
        if paths.get("streets"):
            handles.append(Line2D([0], [0], color="#5f5e59", linewidth=0.8, label="streets"))
        if paths.get("boundary"):
            handles.append(Line2D([0], [0], color=INK2, linewidth=1.1, linestyle=(0, (5, 3)), label="study-area boundary"))
    return handles


_VECTOR_CACHE: dict = {}
_STATE: dict = {"web_failed": False, "reachable": {}, "last": None}


def reset_basemap_state() -> None:
    """Forget a previous tile failure / connectivity result / last choice (a new run, or tests)."""
    _STATE.update(web_failed=False, reachable={}, last=None)
    cartography.reset()


def last_basemap():
    """What the most recent map was drawn on: 'web', 'vectors' or None (plain background)."""
    return _STATE["last"]


def _vector_basemap(ax, cfg: dict, quiet: bool = False, strict: bool = False) -> bool:
    """Offline basemap from the layers the pipeline already uses: OSM street centrelines (paths.streets) and
    the study-area boundary (paths.boundary), reprojected to data.crs and clipped to the map window.
    `quiet` (used inside the fallback chain) suppresses the warning when nothing is configured or readable;
    `strict` turns a missing / unreadable / empty layer into an error."""
    import geopandas as gpd

    paths, crs = cfg["data"]["paths"], cfg["data"]["crs"]
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    drew = False
    # Drawn ABOVE the prediction cells (zorder 3, under the 35 ug/m3 outline at 4): the cells are opaque, so a layer
    # underneath them would be invisible. Web tiles are different: they sit below and the cells turn translucent.
    for key, style in (("streets", {"color": "#5f5e59", "linewidth": 0.45, "alpha": 0.85}),
                       ("boundary", {"color": INK2, "linewidth": 1.1, "linestyle": (0, (5, 3))})):    # dashed: not the 35 ug/m3 outline
        if not paths.get(key):
            continue
        try:
            if (paths[key], crs) not in _VECTOR_CACHE:
                _VECTOR_CACHE[(paths[key], crs)] = gpd.read_file(paths[key]).to_crs(crs)
        except Exception as exc:                                  # missing / unreadable file, no CRS
            if strict:
                raise ValueError(f"vector basemap layer '{key}' could not be read: {exc}") from exc
            if not quiet:
                warnings.warn(f"vector basemap layer '{key}' could not be read ({exc})")
            continue
        layer = _VECTOR_CACHE[(paths[key], crs)].cx[left:right, bottom:top]
        if layer.empty:
            continue
        (layer.boundary if layer.geom_type.str.contains("Polygon").all() else layer).plot(ax=ax, zorder=3, **style)
        drew = True
    if not drew and strict:
        raise ValueError("output.basemap.strict: no vector layer could be drawn (set data.paths.streets and/or "
                         "data.paths.boundary, inside the map window)")
    if not drew and not quiet:
        warnings.warn("basemap 'vectors' needs data.paths.streets and/or data.paths.boundary inside the map window")
    return drew


def _provider_url(provider):
    """The URL template of a provider: an xyzservices TileProvider / dict (`url`), or a plain string."""
    url = provider if isinstance(provider, str) else (provider.get("url") if hasattr(provider, "get") else None)
    return url if isinstance(url, str) and url.startswith(("http://", "https://")) else None


def _host(url: str):
    from urllib.parse import urlparse
    return urlparse(url.replace("{s}", "a").replace("{r}", "")).hostname


def _is_carto(provider) -> bool:
    url = _provider_url(provider)
    return bool(url) and (_host(url) or "").endswith("cartocdn.com")


def _with_key(provider, key: str, param: str):
    """A copy of the provider that sends the API key; the original is never modified.

    An xyzservices provider such as CartoDB.Positron already carries its key placeholder in the URL
    (`...{y}{r}.png?key={apikey}`), so it is filled the xyzservices way: provider(apikey=key). Any other provider
    gets `?param=key` appended to its URL."""
    url = _provider_url(provider)
    if "{apikey}" in url and callable(provider):
        return provider(apikey=key)
    keyed = url + ("&" if "?" in url else "?") + f"{param}={key}"
    if callable(provider):                                      # xyzservices.TileProvider without the placeholder
        return provider(url=keyed)
    return {**provider, "url": keyed}                           # a plain dict


def _redact(text: str, secrets) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def _reachable(provider) -> bool:
    """Quick connectivity test (2 s) for the tile host, remembered per host, so an offline machine does not
    wait for a network timeout on every map. Providers without a url (e.g. test doubles) are assumed reachable."""
    import socket

    url = _provider_url(provider)
    if not url:
        return True
    host = _host(url)
    if host not in _STATE["reachable"]:
        try:
            socket.create_connection((host, 443), timeout=2).close()
            _STATE["reachable"][host] = True
        except OSError:
            _STATE["reachable"][host] = False
    return _STATE["reachable"][host]


def _no_key(strict: bool, message: str) -> bool:
    """A tile source that needs a key has none: never request it (CARTO would answer with watermarked tiles)."""
    if strict:
        raise RuntimeError(f"output.basemap.strict: {message}")
    _STATE["web_failed"] = "nokey"
    warnings.warn(f"{message}; no web-tile basemap for this run")
    return False


def _resolve_source(cx, o: dict):
    """(source, attribution, secrets) for contextily, or (None, None, message) when a required key is missing.

    `basemap_url` (a URL template; ${NAME} filled from the environment) wins over `basemap_provider`; a CARTO
    provider gets the key from the environment variable `basemap_key_env` as ?<carto_key_param>=<key>."""
    import os
    import re

    template = o.get("basemap_url")
    if template:
        names = re.findall(r"\$\{(\w+)\}", template)
        missing = [n for n in names if not os.environ.get(n)]
        if missing:
            return None, None, f"basemap_url needs {', '.join(missing)}: set it in model/.env (see .env.example)"
        secrets = [os.environ[n] for n in names]
        for n in names:
            template = template.replace("${" + n + "}", os.environ[n])
        return template, o.get("basemap_attribution"), secrets

    provider = cx.providers
    for part in o["basemap_provider"].split("."):
        provider = provider[part]
    if _is_carto(provider):
        env = o.get("basemap_key_env", "CARTO_API_KEY")
        key = os.environ.get(env)
        if not key:
            return None, None, (f"CARTO basemap tiles need a free API key: set {env} in model/.env (copy .env.example; "
                                "request a key at https://www.carto.com/basemaps/apikey/). Without it CARTO serves "
                                "watermarked tiles, so none are requested")
        return _with_key(provider, key, o.get("carto_key_param", "key")), None, [key]
    return provider, None, []


def _web_basemap(ax, crs, cfg: dict, strict: bool) -> bool:
    """Web tiles through contextily. Once they have failed (not installed, offline, no key, blocked) later maps of
    the same run skip them instead of failing again. `strict` (output.basemap.strict) raises instead of falling
    back. API keys are never printed: messages and strict errors have them replaced by ***."""
    from config_parser import resolve_path

    failed = _STATE["web_failed"]                  # False | "import" (library missing) | "network" | "nokey"
    if failed in ("network", "nokey") or (failed == "import" and not strict):
        return False
    try:
        import contextily as cx
    except ImportError:
        if strict:
            raise                                  # strict: the library is required
        _STATE["web_failed"] = "import"
        warnings.warn("contextily is not installed: no web-tile basemap (pip install contextily; tiles need internet)")
        return False

    o, secrets = cfg["output"], []
    try:
        source, attribution, extra = _resolve_source(cx, o)
        if source is None:
            return _no_key(strict, extra)
        secrets = extra
        if not _reachable(source):
            if strict:
                raise ConnectionError("output.basemap.strict: the tile server is not reachable (offline?)")
            _STATE["web_failed"] = "network"
            warnings.warn("tile server not reachable (offline?): no web-tile basemap for this run")
            return False
        cache = o.get("basemap_cache_dir")
        if cache and hasattr(cx, "set_cache_dir"):                 # tiles fetched once can be reused offline later
            cx.set_cache_dir(resolve_path(cache))
        extra_args = {"attribution": attribution} if isinstance(source, str) and attribution else {}
        cx.add_basemap(ax, crs=str(crs), source=source, zorder=0, **extra_args)   # draws the provider's attribution
        return True
    except Exception as exc:                                    # blocked, unknown provider, tile error
        message = str(exc)
        if strict:
            if any(secret and secret in message for secret in secrets):
                raise RuntimeError(_redact(message, secrets)) from None      # no traceback chain that shows the URL
            raise                                               # (also re-raises the strict "unreachable" error above)
        _STATE["web_failed"] = "network"
        warnings.warn(f"basemap could not be loaded ({_redact(message, secrets)}); drawing without web tiles")
        return False


_LEGACY_MODES = {                      # the old single-string setting, mapped onto the switches
    "auto": {"enabled": True, "web_tiles": True, "vector_layers": True, "strict": False},
    "contextily": {"enabled": True, "web_tiles": True, "vector_layers": False, "strict": True},
    "vectors": {"enabled": True, "web_tiles": False, "vector_layers": True, "strict": False},
    "none": {"enabled": False, "web_tiles": False, "vector_layers": False, "strict": False},
}


def basemap_settings(cfg: dict) -> dict:
    """The four basemap switches (enabled, web_tiles, vector_layers, strict) from output.basemap, which is a
    settings block (missing keys default to the first three on, strict off) or a legacy mode string."""
    b = cfg["output"]["basemap"]
    if isinstance(b, str):
        return dict(_LEGACY_MODES[b])
    return {"enabled": True, "web_tiles": True, "vector_layers": True, "strict": False, **b}


def _add_basemap(ax, crs, cfg: dict):
    """Pick the background of one map and remember the choice. Returns 'web', 'vectors' or None.

    Order when everything is on: web tiles -> offline vector layers -> plain. Each switch removes a step
    (output.basemap.web_tiles / vector_layers / enabled); without `strict` an unavailable step is skipped with
    a notice, with `strict` it raises."""
    s = basemap_settings(cfg)
    kind = None
    if s["enabled"]:
        if not (s["web_tiles"] or s["vector_layers"]):
            warnings.warn("output.basemap.enabled is true but web_tiles and vector_layers are both off: plain background")
        if s["web_tiles"] and _web_basemap(ax, crs, cfg, strict=s["strict"]):
            kind = "web"
        elif s["vector_layers"] and _vector_basemap(ax, cfg, quiet=(s["web_tiles"] and not s["strict"]),
                                                    strict=s["strict"]):
            kind = "vectors"
    _STATE["last"] = kind
    return kind


STYLES = ("aqi", "continuous")
UNIT = "PM2.5 (µg/m³)"
NOTE_PM25 = ("PM2.5 = predicted hourly fine particulate matter in µg/m³, one value per grid cell. Cells = grid cells inside the NCR "
             "that have a prediction; coverage = their share of all cells inside the NCR.")
NOTE_HISTORY = ("A cell without a sensor takes the PM2.5 of its 5 nearest training stations as its recent history, together with its "
                "320 m morphology patch. White dots are the monitoring stations.")
NOTE_AQI = ("Colours are the Philippine AQI bands (DENR DAO 2020-14); only the Good, Fair and Unhealthy-for-sensitive-groups limits "
            "are confirmed (configs/output.yaml).")
NOTE_CONTINUOUS = ("Continuous colour scale, the same for every panel of the day, from the 2nd to the 98th percentile of the "
                   "predictions inside the NCR (values beyond are shown in the end colours).")


def view_extent(extent: tuple, cfg: dict) -> tuple:
    """The part of the raster to show: the whole grid, or with a boundary the NCR's bounding box (plus a small margin) where it
    overlaps the grid. (left, right, bottom, top)"""
    boundary = cartography.active_boundary(cfg)
    left, right, bottom, top = extent
    if boundary is None:
        return extent
    minx, miny, maxx, maxy = boundary.total_bounds
    mx, my = 0.01 * (maxx - minx), 0.01 * (maxy - miny)
    return (max(left, minx - mx), min(right, maxx + mx), max(bottom, miny - my), min(top, maxy + my))


def xgboost_look(cfg: dict, style: str | None = None) -> bool:
    """output.maps.look == 'xgboost': the maps (AQI and continuous) are drawn like the XGBoost branch's (xgboost/maps/surfaces.py), so
    the two models' figures can be compared side by side. `style` is accepted for the callers that have one."""
    return cartography.settings(cfg)["look"] == "xgboost"


def style_colourbar(cbar, image, cfg: dict) -> None:
    """XGBoost look: colourbar numbers every 5 ug/m3 inside the scale (0, 5, 10, 15, 20 on a 0-24 scale), as on the XGBoost maps.
    Too short a scale for two ticks keeps the automatic ones."""
    if not xgboost_look(cfg):
        return
    vmin, vmax = image.norm.vmin, image.norm.vmax
    ticks = np.arange(np.ceil(vmin / 5.0) * 5.0, vmax + 1e-9, 5.0)
    if len(ticks) >= 2:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:g}" for t in ticks])


def continuous_range(arrays, transform, cfg: dict) -> tuple:
    """(vmin, vmax) of the continuous colour scale: output.maps.continuous_range = [lo, hi], or 'auto' = the 2nd and 98th percentile of
    the predictions inside the NCR over all `arrays` (so every panel of a day shares one scale)."""
    fixed = cartography.settings(cfg)["continuous_range"]
    if fixed != "auto":
        return float(fixed[0]), float(fixed[1])
    boundary = cartography.active_boundary(cfg)
    vals = []
    for a in arrays:
        a = np.asarray(a, dtype=float)
        if boundary is not None:
            a = np.where(cartography.inside_mask(transform, a.shape, boundary), a, np.nan)
        vals.append(a[np.isfinite(a)])
    v = np.concatenate(vals) if vals else np.array([])
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [2, 98])
    if xgboost_look(cfg):                                  # XGBoost's rule: a lone map starts at 0, a set shares [max(0, p2), p98]
        lo = 0.0 if len(arrays) == 1 else max(0.0, float(lo))
    return (float(lo), float(hi)) if hi > lo else (float(lo) - 0.5, float(hi) + 0.5)


def draw_surface(ax, values: np.ndarray, transform, crs, cfg: dict, threshold=None, stations=None, title=None, *,
                 style: str = "aqi", vrange=None, single: bool = False):
    """One snapshot on `ax`: AQI-coloured cells (style 'aqi') or a continuous colour scale (style 'continuous'), clipped to the NCR
    when a boundary is set, with city outlines and names, scale bar and north arrow (output.maps), the optional outline of cells above
    `threshold` (output.outline_above_threshold) and the stations. The title gets '(N cells, coverage C)'. Returns the image drawn
    (for a colourbar)."""
    if style not in STYLES:
        raise ValueError(f"map style must be one of {STYLES}, got {style!r}")
    bins = cfg["output"]["aqi"]["bins"]
    left, right, bottom, top = extent = extent_of(transform, values.shape)
    vl, vr, vb, vt = view_extent(extent, cfg)
    ax.set_xlim(vl, vr)
    ax.set_ylim(vb, vt)
    cartography.fill_no_data(ax, cfg)
    based = _add_basemap(ax, crs, cfg) == "web"                  # tiles are busy: let them show through the cells

    xgb = xgboost_look(cfg, style)                               # XGBoost look: the raw cells (no smoothing), the same see-through surface
    shown = cartography.clip_display(np.asarray(values, dtype=np.float64) if xgb else display_grid(values, cfg),
                                     (left, right, bottom, top), cfg)    # smoothed (or the raw cells), same area either way
    alpha = (0.62 if style == "continuous" else 1.0) if xgb else 0.72 if based else 1.0     # XGBoost: see-through continuous, solid AQI cells
    if style == "aqi":
        cmap = ListedColormap([b["color"] for b in bins]).with_extremes(bad=(0, 0, 0, 0))
        image = ax.imshow(np.ma.masked_invalid(categorize(shown, cfg)), extent=extent, origin="upper", cmap=cmap,
                          vmin=-0.5, vmax=len(bins) - 0.5, interpolation="nearest", alpha=alpha, zorder=2)
    else:
        vmin, vmax = vrange if vrange is not None else continuous_range([values], transform, cfg)
        cmap = plt.get_cmap(cartography.settings(cfg)["continuous_cmap"]).with_extremes(bad=(0, 0, 0, 0))
        image = ax.imshow(np.ma.masked_invalid(shown), extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation="nearest", alpha=alpha, zorder=2)

    if threshold is not None and _outline_on(cfg):
        above = np.where(np.isfinite(shown), shown > threshold, False)
        if above.any() and not above.all():
            xs = left + (np.arange(shown.shape[1]) + 0.5) * (right - left) / shown.shape[1]
            ys = top - (np.arange(shown.shape[0]) + 0.5) * (top - bottom) / shown.shape[0]
            ax.contour(xs, ys, above.astype(float), levels=[0.5], colors=[INK], linewidths=1.1, zorder=4)
    if stations is not None and len(stations):
        ax.scatter(stations["x"], stations["y"], s=(22 if style == "aqi" else 12) if xgb else 14, c=SURFACE,
                   edgecolors=INK, linewidths=(0.9 if style == "aqi" else 0.5) if xgb else 0.8, zorder=5)
    ax.set_xlim(vl, vr)
    ax.set_ylim(vb, vt)
    ax.set_aspect("equal")
    cartography.km_axes(ax)
    cartography.furniture(ax, cfg)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xgb:
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_fontsize(7)
        ax.yaxis.label.set_fontsize(7)
    if title:
        s = cartography.grid_stats(values, transform, cfg)
        if xgb and single:                                       # XGBoost's two-line title: what is drawn, then the cells
            head = title if title.startswith("PM2.5") else f"PM2.5 {title}"
            if style == "continuous":
                ax.set_title(f"{head} — continuous scale, {vmin:.1f}-{vmax:.1f} µg/m³\n"
                             f"{s['cells']:,} cells · colour is comparable across snapshots", color=INK, fontsize=9, loc="left")
            else:
                limit = f" · outline = above {threshold:g} µg/m³" if threshold is not None else ""
                ax.set_title(f"{head} — {abs(transform.a):g} m cells\n{s['cells']:,} usable cells · coverage {s['coverage']:.2f}{limit}",
                             color=INK, fontsize=9, loc="left")
        else:
            sep = "\n" if len(title) > 22 else "   "              # a long title would run past the map: put the counts on a second line
            ax.set_title(f"{title}{sep}({s['cells']:,} cells, coverage {s['coverage']:.2f})", color=INK, fontsize=10, loc="left")
    return image


def _stations(cfg: dict, stations):
    return stations if cfg["output"]["show_stations"] else None


def _notes(style: str, cfg: dict | None = None) -> list[str]:
    note = NOTE_CONTINUOUS
    fixed = cartography.settings(cfg)["continuous_range"] if cfg is not None else "auto"
    if style != "aqi" and fixed != "auto":
        note = (f"Continuous colour scale, fixed at {float(fixed[0]):g} to {float(fixed[1]):g} µg/m³ and shared by every map "
                "(and by the XGBoost maps), so colours compare directly; values beyond are shown in the end colours.")
    return [NOTE_PM25, NOTE_HISTORY, NOTE_AQI if style == "aqi" else note]


def _legend(fig, cfg: dict, style: str, threshold, bottom: float, ncol: int, fontsize: float) -> None:
    """AQI ranges (aqi style) or just the 'No data' swatch (continuous), above the footer."""
    handles = (_legend_entries(cfg, threshold, f"outline: above {threshold} ug/m3") if style == "aqi" else []) + [cartography.no_data_patch()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, bottom), ncol=ncol, fontsize=fontsize, frameon=False,
               labelcolor=INK2)


def render_snapshot(values, transform, crs, path: str, title: str, cfg: dict, threshold, stations=None, *, style: str = "aqi",
                    vrange=None, caption: str | None = None) -> str:
    o = cfg["output"]
    fig, ax = plt.subplots(figsize=(8.5, 7.6) if xgboost_look(cfg, style) else (6.6, 7.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    image = draw_surface(ax, values, transform, crs, cfg, threshold, _stations(cfg, stations), title, style=style, vrange=vrange, single=True)
    if style == "continuous":
        cbar = fig.colorbar(image, ax=ax, shrink=0.7 if xgboost_look(cfg, style) else 0.8, pad=0.02)
        cbar.set_label(UNIT, color=INK2, fontsize=8)
        style_colourbar(cbar, image, cfg)
    foot = cartography.footer(fig, cfg, caption, _notes(style, cfg))
    _legend(fig, cfg, style, threshold, foot + 0.005, 2, 7.5)
    fig.tight_layout(rect=(0, foot + 0.13, 1, 1))
    fig.savefig(path, dpi=o["dpi"], facecolor=SURFACE)
    plt.close(fig)
    return path


def render_panels(stack, labels: list[str], transform, crs, path: str, title: str, cfg: dict, threshold,
                  stations=None, *, style: str = "aqi", vrange=None, caption: str | None = None) -> str:
    """All snapshots of a day side by side (one shared scale: the AQI bands, or one continuous colourbar)."""
    o = cfg["output"]
    n = len(stack)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols + (1.0 if style == "continuous" else 0), 6.0 * nrows + 0.9),
                             facecolor=SURFACE, squeeze=False)
    if style == "continuous" and vrange is None:
        vrange = continuous_range(stack, transform, cfg)
    image = None
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        ax.set_facecolor(SURFACE)
        image = draw_surface(ax, stack[i], transform, crs, cfg, threshold, _stations(cfg, stations), labels[i], style=style, vrange=vrange)
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    foot = cartography.footer(fig, cfg, caption, _notes(style, cfg))
    _legend(fig, cfg, style, threshold, foot + 0.005, 3, 8)
    right = 0.9 if style == "continuous" else 1.0
    bottom = foot + 0.06
    fig.tight_layout(rect=(0, bottom, right, 0.97), h_pad=2.5)
    if style == "continuous" and image is not None:
        cax = fig.add_axes([0.92, bottom + 0.15 * (0.97 - bottom), 0.018, 0.6 * (0.97 - bottom)])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label(UNIT, color=INK2, fontsize=9)
        style_colourbar(cbar, image, cfg)
    fig.savefig(path, dpi=o["dpi"], facecolor=SURFACE)
    plt.close(fig)
    return path


FILE_STEMS = {"aqi": ("aqi_{h}", "daily_panels", "daily_mean"),                       # file stems per style (the AQI names are the old ones)
              "continuous": ("continuous_{h}", "continuous_panels", "continuous_daily_mean")}


def render_day(stack, labels: list[str], transform, crs, maps_dir, date, cfg: dict, stations, caption: str | None, mean=None) -> None:
    """All the PNG maps of one date (output.maps.styles): an hourly map per snapshot, the panels, and the daily mean when `mean` is given.
    `stack` = predictions [hours, rows, cols]. Used by --stage infer and to redraw old results from their prediction_volume.tif."""
    from pathlib import Path

    o = cfg["output"]
    maps_dir = Path(maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    styles = cartography.settings(cfg)["styles"]
    shared = continuous_range(stack, transform, cfg) if "continuous" in styles else None      # one colour scale for all hours of the day
    for style in styles:
        hourly_stem, panels_stem, mean_stem = FILE_STEMS[style]
        vr = shared if style == "continuous" else None
        for i, label in enumerate(labels):
            render_snapshot(stack[i], transform, crs, str(maps_dir / f"{hourly_stem.format(h=label.replace(':', ''))}.png"),
                            f"PM2.5 prediction, {date:%Y-%m-%d} {label}", cfg, o["highlight_threshold"], stations, style=style,
                            vrange=vr, caption=caption)
        if len(labels) > 1:
            render_panels(stack, labels, transform, crs, str(maps_dir / f"{panels_stem}.png"), f"PM2.5 prediction, {date:%Y-%m-%d}", cfg,
                          o["highlight_threshold"], stations, style=style, vrange=vr, caption=caption)
            if mean is not None:
                render_snapshot(mean, transform, crs, str(maps_dir / f"{mean_stem}.png"),
                                f"24-hour mean of {len(labels)} snapshots, {date:%Y-%m-%d}", cfg, o["daily_guideline"], stations,
                                style=style, caption=caption)
