"""Figures of the feature-group Grad-CAM pass: per band an NCR map of its high and low ends with the hot and cold cells of each end, and
a chart of the predicted PM2.5 in the high against the low end of every band."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explain.bands import meaning_of, unit_of  # noqa: E402
from explain.gradcam_plots import COLD, HOT  # noqa: E402
from output import aqi_maps, cartography  # noqa: E402
from reporting.plotters import INK, INK2, SURFACE  # noqa: E402

HIGH_C, LOW_C = "#6a3d9a", "#2aa876"
NOTE_MAP = ("Purple = the NCR cells where this band is high (at or above the upper limit), green = where it is low (at or below the lower "
            "limit); limits are percentiles of the NCR, taken from the raster. Circles mark picks in the high end, diamonds in the low end; "
            "red = HOT (highest predicted PM2.5 of that end), blue = COLD (lowest). The numbers are the rows of the matching close-up figure.")
NOTE_CHART = ("Bars: mean predicted PM2.5 of all cells of the high end (purple) and the low end (green) of each band, whiskers one standard "
              "deviation across those cells; the dashed line is the NCR mean. It describes the model's predictions where a feature is high "
              "or low, not the effect of changing the feature. Small groups and cells that share a value at the limit are listed in the CSV.")


def plot_group_map(grid: np.ndarray, extent: tuple, transform, band: str, level: str, limits: tuple, picks: list[dict], cfg: dict,
                   path: str, *, stations=None, caption: str | None = None, dpi: int = 150, heading: str | None = None) -> str:
    """`grid`: the band's patch mean on a north-up grid with `extent` (left, right, bottom, top); `limits` = (low, high);
    `picks`: dicts with side ('high' or 'low'), kind ('hot' or 'cold'), number, x, y."""
    low, high = limits
    boundary = cartography.active_boundary(cfg)
    inside = cartography.inside_mask(transform, grid.shape, boundary) if boundary is not None else np.ones(grid.shape, bool)
    with np.errstate(invalid="ignore"):
        is_high, is_low = (grid >= high) & inside, (grid <= low) & inside
    rgba = np.zeros(grid.shape + (4,), dtype="float32")
    for mask, color in ((is_high, HIGH_C), (is_low, LOW_C)):
        rgba[mask] = matplotlib.colors.to_rgba(color, 0.78)
    fig, ax = plt.subplots(figsize=(6.8, 8.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    vl, vr, vb, vt = aqi_maps.view_extent(extent, cfg)
    ax.set_xlim(vl, vr)
    ax.set_ylim(vb, vt)
    cartography.fill_no_data(ax, cfg)
    ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest", zorder=2)
    for side, marker in (("high", "o"), ("low", "D")):
        for kind, color in (("hot", HOT), ("cold", COLD)):
            pts = [p for p in picks if p["side"] == side and p["kind"] == kind]
            if not pts:
                continue
            ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=105, marker=marker, facecolor=color, edgecolor="white",
                       linewidth=1.3, zorder=9)
            for p in pts:
                ax.text(p["x"], p["y"], str(p["number"]), fontsize=6, color="white", ha="center", va="center", zorder=10,
                        fontweight="bold", path_effects=[pe.withStroke(linewidth=0.6, foreground=color)])
    if stations is not None and len(stations) and cfg["output"]["show_stations"]:
        ax.scatter(stations["x"], stations["y"], s=6, c=SURFACE, edgecolors=INK, linewidths=0.4, zorder=5)
    ax.set_xlim(vl, vr)
    ax.set_ylim(vb, vt)
    ax.set_aspect("equal")
    cartography.km_axes(ax)
    cartography.furniture(ax, cfg)
    ax.tick_params(labelsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    unit = unit_of(band)
    handles = [Patch(facecolor=HIGH_C, alpha=0.78, label=f"high: >= {high:.3g} {unit}"),
               Patch(facecolor=LOW_C, alpha=0.78, label=f"low: <= {low:.3g} {unit}"), cartography.no_data_patch(),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=HOT, markeredgecolor="white", markersize=9, label="hot (circle = high end)"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=COLD, markeredgecolor="white", markersize=9, label="cold (circle = high end)"),
               Line2D([0], [0], marker="D", color="none", markerfacecolor=HOT, markeredgecolor="white", markersize=8, label="hot (diamond = low end)"),
               Line2D([0], [0], marker="D", color="none", markerfacecolor=COLD, markeredgecolor="white", markersize=8, label="cold (diamond = low end)")]
    foot = cartography.footer(fig, cfg, caption, (NOTE_MAP, f"{band.replace('_', ' ')}: {meaning_of(band)}; patch mean over 320 m."))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, foot + 0.005), ncol=3, fontsize=7, frameon=False, labelcolor=INK2)
    fig.suptitle("\n".join(textwrap.wrap(heading or f"{band}: high and low ends ({level})", 90)), color=INK, fontsize=9.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, foot + 0.1, 1, 0.94))
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_group_comparison(table, ncr_mean: float, path: str, *, cfg: dict | None = None, caption: str | None = None, dpi: int = 150,
                          heading: str | None = None) -> str:
    """`table` (feature_group_summary.csv): band, level, side, cells, mean_prediction, sd_prediction. One panel per level."""
    levels = list(dict.fromkeys(table["level"]))
    bands = list(dict.fromkeys(table["band"]))
    fig, axes = plt.subplots(1, len(levels), figsize=(5.4 * len(levels) + 2.2, 0.8 * len(bands) + 3.0), facecolor=SURFACE, squeeze=False,
                             sharey=True)
    ypos = np.arange(len(bands))[::-1]
    h = 0.34
    for ax, level in zip(axes[0], levels):
        ax.set_facecolor(SURFACE)
        for k, (side, color) in enumerate((("high", HIGH_C), ("low", LOW_C))):
            means, sds = [], []
            for band in bands:
                row = table[(table["level"] == level) & (table["band"] == band) & (table["side"] == side)]
                means.append(float(row["mean_prediction"].iloc[0]) if len(row) else np.nan)
                sds.append(float(row["sd_prediction"].iloc[0]) if len(row) else 0.0)
            ax.barh(ypos + (0.5 - k) * h, means, height=h, xerr=np.nan_to_num(sds), color=color, ecolor=INK2,
                    error_kw=dict(lw=0.9, capsize=2), zorder=3)
        ax.axvline(ncr_mean, color=INK, linestyle="--", linewidth=1, zorder=4)
        ax.set_title(level, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("predicted PM2.5 (µg/m³)", fontsize=8, color=INK2)
        ax.grid(axis="x", color="#e6e5e1", zorder=0)
        ax.tick_params(colors=INK2, labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_yticks(ypos)
    axes[0][0].set_yticklabels([b.replace("_", " ") + f"\n({unit_of(b)})" for b in bands], fontsize=7, color=INK)
    fig.legend(handles=[Patch(facecolor=HIGH_C, label="high end of the band"), Patch(facecolor=LOW_C, label="low end of the band"),
                        Line2D([0], [0], color=INK, linestyle="--", label=f"NCR mean {ncr_mean:.1f}")],
               loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.suptitle(heading or "Predicted PM2.5 where each band is high and low", color=INK, fontsize=10, x=0.02, ha="left")
    foot = cartography.footer(fig, cfg, caption, (NOTE_CHART,)) if cfg is not None else 0.0
    fig.tight_layout(rect=(0, foot, 1, 0.93))
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path
