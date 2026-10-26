"""Map furniture shared by every figure (prediction maps, Grad-CAM map, morphology sheet): the NCR boundary and the clip to it,
city outlines and names, scale bar, north arrow, kilometre axes, a footer with notes and a caption.

Everything here is DISPLAY only, like the smoothing in aqi_maps: GeoTIFFs and the numbers in the summaries keep every cell.
The boundary file is data.paths.boundary (python -m preprocessing.boundaries). Without it (unset, missing or unreadable) one warning
is given and the maps are simply not clipped and have no city labels.
"""
import textwrap
import warnings

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

from reporting.plotters import INK, INK2, SURFACE

NO_DATA = "#d9d9d9"
# metres (dx, dy) and alignment for the labels of small, adjacent cities that would overlap at the centre of their polygon
LABEL_NUDGE = {"Navotas": (-900, -500, "right"), "Malabon": (500, 500, "left"), "San Juan": (-300, 450, "right"),
               "Pateros": (900, -300, "left")}
DEFAULTS = {"styles": ["aqi", "continuous"], "clip_to_boundary": True, "city_labels": True, "scale_bar": True,
            "north_arrow": True, "caption": True, "metric_notes": True, "continuous_cmap": "viridis", "continuous_range": "auto"}
_CACHE: dict = {}


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg["output"].get("maps") or {})}


def reset() -> None:
    """Forget the cached boundary (a new run, or tests)."""
    _CACHE.clear()


def load_boundary(cfg: dict):
    """The NCR city polygons in data.crs (columns city, geometry), or None. A problem is reported once."""
    path = cfg["data"]["paths"].get("boundary")
    if not path:
        return None
    key = (str(path), cfg["data"]["crs"])
    if key not in _CACHE:
        try:
            import geopandas as gpd

            gdf = gpd.read_file(path).to_crs(cfg["data"]["crs"])
            if "city" not in gdf.columns:
                gdf["city"] = ""
            _CACHE[key] = gdf
        except Exception as exc:                          # missing file, unreadable, no CRS
            warnings.warn(f"NCR boundary {path} could not be read ({exc}): the maps are not clipped to the NCR and have no city labels")
            _CACHE[key] = None
    return _CACHE[key]


def active_boundary(cfg: dict):
    """The boundary when the maps are set to use it, else None."""
    return load_boundary(cfg) if settings(cfg)["clip_to_boundary"] else None


def inside_mask(transform, shape, boundary) -> np.ndarray:
    """True for the cells of the grid (north-up `transform`, `shape` rows x cols) that touch the boundary."""
    from rasterio.features import geometry_mask

    return geometry_mask(list(boundary.geometry), out_shape=tuple(shape), transform=transform, invert=True, all_touched=True)


def grid_stats(values: np.ndarray, transform, cfg: dict) -> dict:
    """Cells with a prediction and the share of the cells inside the NCR that have one (of all cells without a boundary)."""
    boundary = active_boundary(cfg)
    finite = np.isfinite(values)
    if boundary is None:
        return {"cells": int(finite.sum()), "coverage": float(finite.mean()) if finite.size else 0.0, "clipped": False}
    inside = inside_mask(transform, values.shape, boundary)
    n = int(inside.sum())
    cells = int((finite & inside).sum())
    return {"cells": cells, "coverage": cells / n if n else 0.0, "clipped": True}


def clip_display(shown: np.ndarray, extent: tuple, cfg: dict) -> np.ndarray:
    """NaN outside the NCR on the (possibly smoothed) grid that is drawn. `extent` = (left, right, bottom, top)."""
    boundary = active_boundary(cfg)
    if boundary is None:
        return shown
    from rasterio.transform import from_bounds

    left, right, bottom, top = extent
    inside = inside_mask(from_bounds(left, bottom, right, top, shown.shape[1], shown.shape[0]), shown.shape, boundary)
    return np.where(inside, shown, np.nan)


def fill_no_data(ax, cfg: dict) -> None:
    """Grey underlay for the whole NCR, so a cell without a prediction shows as 'No data' (the legend swatch) and not as blank paper."""
    boundary = active_boundary(cfg)
    if boundary is not None:
        boundary.plot(ax=ax, facecolor=NO_DATA, edgecolor="none", zorder=1)


def draw_cities(ax, cfg: dict, zorder: int = 6, labels: bool | None = None) -> None:
    """City outlines, and their names when output.maps.city_labels (or `labels`), inside the current view."""
    boundary = active_boundary(cfg)
    if boundary is None:
        return
    boundary.boundary.plot(ax=ax, color=INK2, linewidth=0.7, zorder=zorder)
    if not (settings(cfg)["city_labels"] if labels is None else labels):
        return
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for name, pt in zip(boundary["city"], boundary.geometry.representative_point()):
        dx, dy, ha = LABEL_NUDGE.get(name, (0, 0, "center"))
        if name and min(x0, x1) <= pt.x <= max(x0, x1) and min(y0, y1) <= pt.y <= max(y0, y1):
            ax.text(pt.x + dx, pt.y + dy, name, fontsize=6.5, color=INK, ha=ha, va="center", zorder=zorder + 1,
                    path_effects=[pe.withStroke(linewidth=2, foreground=SURFACE)])


def nice_length(target_m: float) -> float:
    """The largest 1, 2 or 5 x 10^k metres that does not exceed target_m."""
    if target_m <= 0:
        return 1.0
    exp = 10 ** np.floor(np.log10(target_m))
    return float(max(m * exp for m in (1, 2, 5) if m * exp <= target_m))


def scale_bar(ax, zorder: int = 8) -> float:
    """A two-tone bar about a quarter of the map width, bottom left, labelled in km (m below 1 km). Returns its length in m."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    length = nice_length(abs(x1 - x0) * 0.25)
    left, base = min(x0, x1) + abs(x1 - x0) * 0.05, min(y0, y1) + abs(y1 - y0) * 0.10
    h = abs(y1 - y0) * 0.008
    for i, face in enumerate((INK, SURFACE)):
        ax.add_patch(Rectangle((left + i * length / 2, base), length / 2, h, facecolor=face, edgecolor=INK, linewidth=0.7, zorder=zorder))
    unit, div = ("km", 1000.0) if length >= 1000 else ("m", 1.0)
    halo = [pe.withStroke(linewidth=2, foreground=SURFACE)]
    for frac, text in ((0, "0"), (0.5, f"{length / 2 / div:g}"), (1, f"{length / div:g} {unit}")):
        ax.text(left + frac * length, base + h * 1.8, text, fontsize=6.5, ha="center", va="bottom", color=INK, zorder=zorder, path_effects=halo)
    return length


def north_arrow(ax, zorder: int = 8) -> None:
    ax.annotate("", xy=(0.92, 0.16), xytext=(0.92, 0.07), xycoords="axes fraction", zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, shrinkA=0, shrinkB=0))
    ax.text(0.92, 0.17, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold", color=INK, zorder=zorder,
            path_effects=[pe.withStroke(linewidth=2, foreground=SURFACE)])


def km_axes(ax) -> None:
    """Easting / northing in kilometres, as on the reference maps (the CRS is metres)."""
    to_km = FuncFormatter(lambda v, _: f"{v / 1000:g}")
    ax.xaxis.set_major_formatter(to_km)
    ax.yaxis.set_major_formatter(to_km)
    ax.set_xlabel("easting (km)", fontsize=7, color=INK2)
    ax.set_ylabel("northing (km)", fontsize=7, color=INK2)
    ax.tick_params(colors=INK2, labelsize=7)


def furniture(ax, cfg: dict, labels: bool | None = None) -> None:
    """City outlines + names, scale bar and north arrow, as the settings ask. Call after the axes limits are final."""
    s = settings(cfg)
    draw_cities(ax, cfg, labels=labels)
    if s["scale_bar"]:
        scale_bar(ax)
    if s["north_arrow"]:
        north_arrow(ax)


def no_data_patch() -> Patch:
    return Patch(facecolor=NO_DATA, edgecolor=INK2, linewidth=0.5, label="No data")


def footer(fig, cfg: dict, caption: str | None = None, notes=()) -> float:
    """Metric notes (what each figure element means) and a caption at the bottom of the figure. Returns the fraction of the figure
    height they use, so the caller can reserve it: fig.tight_layout(rect=(0, frac + ..., 1, 1))."""
    s = settings(cfg)
    chars = max(40, int(fig.get_figwidth() * 17))
    lines = []
    if s["metric_notes"]:
        for note in notes:
            lines += textwrap.wrap(note, chars, initial_indent="• ", subsequent_indent="   ")
    if s["caption"] and caption:
        lines += ([""] if lines else []) + textwrap.wrap(caption, chars)
    if not lines:
        return 0.0
    height_in = (len(lines) * 6.5 * 1.35) / 72 + 0.12
    fig.text(0.012, 0.006, "\n".join(lines), fontsize=6.5, color=INK2, ha="left", va="bottom", linespacing=1.3)
    return height_in / fig.get_figheight()
