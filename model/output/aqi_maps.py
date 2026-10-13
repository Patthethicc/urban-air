"""Prediction surfaces as maps on the Philippine AQI scale (Output.txt).

Cells above the highlight threshold (35 ug/m3) are outlined. A basemap of the NCR is drawn when
`contextily` is installed (output.yaml basemap: auto); otherwise the map keeps a plain background.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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


def _legend_entries(cfg: dict, threshold, threshold_label: str):
    bins = cfg["output"]["aqi"]["bins"]
    lo, handles = 0, []
    for b in bins:
        rng = f">{lo:g}" if b["upper"] is None else f"{lo:g}-{b['upper']:g}"
        handles.append(Patch(facecolor=b["color"], edgecolor=INK2, linewidth=0.5, label=f"{b['label']} ({rng})"))
        lo = b["upper"]
    if threshold is not None:
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


def draw_surface(ax, values: np.ndarray, transform, crs, cfg: dict, threshold=None, stations=None, title=None):
    """One snapshot on `ax`: AQI-coloured cells, outline of cells above `threshold`, optional stations."""
    bins = cfg["output"]["aqi"]["bins"]
    left, right, bottom, top = extent = extent_of(transform, values.shape)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    based = _add_basemap(ax, crs, cfg) == "web"                  # tiles are busy: let them show through the cells

    cmap = ListedColormap([b["color"] for b in bins]).with_extremes(bad=(0, 0, 0, 0))
    ax.imshow(np.ma.masked_invalid(categorize(values, cfg)), extent=extent, origin="upper", cmap=cmap,
              vmin=-0.5, vmax=len(bins) - 0.5, interpolation="nearest", alpha=0.72 if based else 1.0, zorder=2)

    if threshold is not None:
        above = np.where(np.isfinite(values), values > threshold, False)
        if above.any() and not above.all():
            xs = left + (np.arange(values.shape[1]) + 0.5) * transform.a
            ys = top + (np.arange(values.shape[0]) + 0.5) * transform.e
            ax.contour(xs, ys, above.astype(float), levels=[0.5], colors=[INK], linewidths=1.1, zorder=4)
    if stations is not None and len(stations):
        ax.scatter(stations["x"], stations["y"], s=14, c=SURFACE, edgecolors=INK, linewidths=0.8, zorder=5)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(colors=INK2, labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def _stations(cfg: dict, stations):
    return stations if cfg["output"]["show_stations"] else None


def render_snapshot(values, transform, crs, path: str, title: str, cfg: dict, threshold, stations=None) -> str:
    o = cfg["output"]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    draw_surface(ax, values, transform, crs, cfg, threshold, _stations(cfg, stations), title)
    fig.legend(handles=_legend_entries(cfg, threshold, f"outline: above {threshold} ug/m3"), loc="lower center",
               ncol=2, fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=o["dpi"], facecolor=SURFACE)
    plt.close(fig)
    return path


def render_panels(stack, labels: list[str], transform, crs, path: str, title: str, cfg: dict, threshold,
                  stations=None) -> str:
    """All snapshots of a day side by side (shared AQI scale)."""
    o = cfg["output"]
    n = len(stack)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.6 * nrows + 0.8), facecolor=SURFACE, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        ax.set_facecolor(SURFACE)
        draw_surface(ax, stack[i], transform, crs, cfg, threshold, _stations(cfg, stations), labels[i])
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    fig.legend(handles=_legend_entries(cfg, threshold, f"outline: above {threshold} ug/m3"), loc="lower center",
               ncol=3, fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97), h_pad=2.5)
    fig.savefig(path, dpi=o["dpi"], facecolor=SURFACE)
    plt.close(fig)
    return path
