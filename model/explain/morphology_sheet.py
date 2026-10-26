"""The morphology 'diagnosis' sheet: every band of the Master GeoTIFF mapped over the NCR (like the porosity and green-continuity maps of
an urban diagnosis), in quantile classes with a range legend, with the explained hot and cold cells marked. It shows what the CNN
branch gets to see, so it can be read next to the Grad-CAM figures."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

from explain.bands import cmap_of, meaning_of, title_of, unit_of  # noqa: E402
from explain.gradcam_plots import COLD, HOT  # noqa: E402
from output import aqi_maps, cartography  # noqa: E402
from reporting.plotters import INK, INK2, SURFACE  # noqa: E402


def read_bands(raster_path: str, step_m: float = 50.0):
    """The raster averaged down to about `step_m` cells (NoData ignored): (bands [C, h, w] with NaN for no data, transform, extent).
    The full 10 m grid (10.6 M pixels per band) is never held in memory at once."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling

    with rasterio.open(raster_path) as src:
        k = max(1, int(round(step_m / abs(src.res[0]))))
        h, w = max(1, src.height // k), max(1, src.width // k)
        data = src.read(out_shape=(src.count, h, w), masked=True, resampling=Resampling.average).astype("float64").filled(np.nan)
        transform = src.transform * Affine.scale(src.width / w, src.height / h)
        b = src.bounds
        return data, transform, (b.left, b.right, b.bottom, b.top)


def class_edges(values: np.ndarray, classes: int = 6) -> np.ndarray:
    """Quantile class limits of the finite values (equal share of cells per class); fewer classes when many values coincide (e.g. 0)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.array([0.0, 1.0])
    edges = np.unique(np.quantile(v, np.linspace(0, 1, classes + 1)))
    if len(edges) < 3:                                                    # nearly constant: an even split instead
        edges = np.unique(np.linspace(v.min(), v.max(), classes + 1))
    return edges if len(edges) >= 2 else np.array([edges[0], edges[0] + 1.0])


def plot_morphology_sheet(raster_path: str, cfg: dict, cells: list[dict], path: str, band_names: list[str], *, stations=None,
                          caption: str | None = None, classes: int = 6, step_m: float = 50.0, dpi: int = 150) -> str:
    data, transform, extent = read_bands(raster_path, step_m)
    n = min(len(band_names), data.shape[0])
    boundary = cartography.active_boundary(cfg)
    inside = cartography.inside_mask(transform, data.shape[1:], boundary) if boundary is not None else np.ones(data.shape[1:], bool)
    vl, vr, vb, vt = aqi_maps.view_extent(extent, cfg)
    ncols = 3
    nrows = int(np.ceil((n + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 6.0 * nrows), facecolor=SURFACE, squeeze=False)
    flat = axes.ravel()
    for i in range(n):
        ax, band = flat[i], band_names[i]
        ax.set_facecolor(SURFACE)
        layer = np.where(inside, data[i], np.nan)
        edges = class_edges(layer, classes)
        cmap = plt.get_cmap(cmap_of(band), max(1, len(edges) - 1)).with_extremes(bad=(0, 0, 0, 0))
        ax.set_xlim(vl, vr)
        ax.set_ylim(vb, vt)
        cartography.fill_no_data(ax, cfg)
        im = ax.imshow(np.ma.masked_invalid(layer), extent=extent, origin="upper", cmap=cmap, norm=BoundaryNorm(edges, cmap.N),
                       interpolation="nearest", zorder=2)
        for group, color, marker in (("hot", HOT, "^"), ("cold", COLD, "o")):
            pts = [c for c in cells if c["group"] == group]
            if pts:
                ax.scatter([c["x"] for c in pts], [c["y"] for c in pts], s=34, marker=marker, facecolor=color, edgecolor="white",
                           linewidth=0.8, zorder=9)
        if stations is not None and len(stations) and cfg["output"]["show_stations"]:
            ax.scatter(stations["x"], stations["y"], s=6, c=SURFACE, edgecolors=INK, linewidths=0.4, zorder=5)
        ax.set_xlim(vl, vr)
        ax.set_ylim(vb, vt)
        ax.set_aspect("equal")
        cartography.km_axes(ax)
        cartography.furniture(ax, cfg, labels=False)
        ax.tick_params(labelsize=6)
        ax.set_title(title_of(band), color=INK, fontsize=9, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, ticks=edges)
        cb.ax.set_yticklabels([f"{e:.3g}" for e in edges])
        cb.ax.tick_params(labelsize=6, colors=INK2)
        cb.set_label(unit_of(band), fontsize=6.5, color=INK2)
    notes = [f"Each map is one band of the Master GeoTIFF (the CNN input), averaged to about {step_m:g} m cells and clipped to the NCR. "
             "Classes hold equal numbers of cells (quantiles); the legend gives their limits; grey = no data.",
             "Red triangles = the explained HOT cells (highest predicted PM2.5), blue circles = the COLD cells (lowest); white dots = stations."]
    notes += [f"{b.replace('_', ' ')} ({unit_of(b)}): {meaning_of(b)}." for b in band_names[:n]]
    ax = flat[n]
    ax.axis("off")
    ax.text(0, 1, "Reading the maps", fontsize=9, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    ax.text(0, 0.94, "\n\n".join("\n".join(textwrap.wrap("• " + t, 52, subsequent_indent="  ")) for t in notes), fontsize=7, color=INK2,
            va="top", transform=ax.transAxes, linespacing=1.25)
    for ax in flat[n + 1:]:
        ax.axis("off")
    foot = cartography.footer(fig, cfg, caption, ())
    fig.suptitle("Morphology of the NCR: what the CNN branch sees", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, foot, 1, 0.965))
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path
