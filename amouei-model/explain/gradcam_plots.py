"""Grad-CAM figures (Output.txt): the patch close-ups with the attended region, and the band-attention chart that summarises many cells.

Vocabulary used on the figures (also in docs/README): a HOT cell is one of the highest-predicted PM2.5 cells of the explained hour,
a COLD cell one of the lowest. Grad-CAM shows which pixels of a cell's 320 m morphology patch drove that cell's prediction.
"""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explain.bands import meaning_of, title_of  # noqa: E402
from output import cartography  # noqa: E402
from reporting.plotters import INK, INK2, ORANGE, SURFACE  # noqa: E402

HOT, COLD = "#c0392b", "#2a78d6"
NOTE_HOT_COLD = ("HOT cells are the highest predicted PM2.5 of the explained hour, COLD cells the lowest (numbers match the map). "
                 "Both are ranked inside the NCR when a boundary is set, and no two are closer than output.explain.min_separation_m, "
                 "so each is a different place.")
NOTE_CAM = ("Grad-CAM: for each cell, the last convolutional layer of the CNN is weighted by how strongly it moves that cell's "
            "prediction; bright = the pixels that mattered. It reads 'what in this 320 m patch mattered, given the cell's PM2.5 history'.")
NOTE_ORANGE = ("The orange line encloses the top 20 % of the attention (output.explain.attention_quantile) and is drawn on every band so "
               "you can see what lies under it. A cause needs checking: attention shows where the model looked, not why PM2.5 is high.")
NOTE_PROFILE = ("Bars: mean over the explained cells of (band value on the attended pixels - patch average) / patch standard deviation. "
                "Above 0 the model attends to the higher values of that band (taller, denser, greener, deeper canyons), below 0 to the lower. "
                "Whiskers: one standard deviation across cells.")


def plot_cam_overlays(raw_patches: np.ndarray, cam: np.ndarray, band_names: list[str], titles: list[str],
                      path: str, quantile: float = 0.8, dpi: int = 150, *, cfg: dict | None = None, px_m: float = 10.0,
                      caption: str | None = None, heading: str | None = None, notes: tuple | None = None) -> str:
    """One row per cell: every morphology band (grey, with units in the title) with the attended region outlined in orange, a 100 m
    scale tick, and the Grad-CAM map with its colourbar in the last column. With `cfg`, notes explaining the figure and a caption
    are added at the bottom (`notes` replaces the default hot/cold note, for figures with another selection rule)."""
    n, bands = len(cam), raw_patches.shape[1]
    fig, axes = plt.subplots(n, bands + 1, figsize=(2.3 * (bands + 1) + 0.7, 2.3 * n + 1.1), facecolor=SURFACE, squeeze=False)
    image = None
    for i in range(n):
        level = max(float(np.quantile(cam[i], quantile)), 1e-9)
        for b in range(bands + 1):
            ax = axes[i, b]
            ax.set_facecolor(SURFACE)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(INK2)
            if b < bands:
                ax.imshow(raw_patches[i, b], cmap="Greys", interpolation="nearest")
                if cam[i].max() > 0 and cam[i].min() < level <= cam[i].max():
                    ax.contour(cam[i], levels=[level], colors=[ORANGE], linewidths=1.6)
                if b == 0:                                              # a 100 m tick, bottom left of the first band
                    h = raw_patches.shape[2]
                    ax.plot([0.8, 0.8 + 100.0 / px_m], [h - 1.5, h - 1.5], color=INK, linewidth=2.2, solid_capstyle="butt")
                    ax.text(0.8, h - 2.6, "100 m", fontsize=5.5, color=INK, va="bottom",
                            bbox=dict(boxstyle="square,pad=0.1", facecolor=SURFACE, edgecolor="none", alpha=0.8))
                if i == 0:
                    ax.set_title(title_of(band_names[b]) if b < len(band_names) else f"band {b + 1}", color=INK, fontsize=7)
            else:
                image = ax.imshow(cam[i], cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
                if i == 0:
                    ax.set_title("Grad-CAM\n(relative importance)", color=INK, fontsize=7)
        axes[i, 0].set_ylabel(titles[i], color=INK2, fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
    if heading:
        fig.suptitle(heading, color=INK, fontsize=10, x=0.02, ha="left")
    foot = cartography.footer(fig, cfg, caption, (*(notes if notes is not None else (NOTE_HOT_COLD,)), NOTE_CAM, NOTE_ORANGE)) if cfg is not None else 0.0
    fig.tight_layout(rect=(0, foot, 0.94, 0.95 if heading else 1))
    if image is not None:
        top, bottom = axes[0, -1].get_position().y1, axes[-1, -1].get_position().y0
        cax = fig.add_axes([0.955, bottom + 0.15 * (top - bottom), 0.014, 0.7 * (top - bottom)])
        fig.colorbar(image, cax=cax).ax.tick_params(labelsize=6, colors=INK2)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_band_profile(rows: list[dict], band_names: list[str], path: str, dpi: int = 150, *, cfg: dict | None = None,
                      caption: str | None = None, heading: str | None = None) -> str:
    """Hot against cold: per morphology band, the mean attention in patch-standard-deviation units over all explained cells of each
    group (rows of explain: group, band, std_units), with one standard deviation across cells as whiskers."""
    groups = [("hot", HOT), ("cold", COLD)]
    fig, ax = plt.subplots(figsize=(9.4, 0.9 * len(band_names) + 3.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ypos = np.arange(len(band_names))[::-1]
    h = 0.34
    for k, (group, color) in enumerate(groups):
        means, sds = [], []
        for band in band_names:
            v = np.array([r["std_units"] for r in rows if r["group"] == group and r["band"] == band], dtype=float)
            means.append(float(v.mean()) if v.size else np.nan)
            sds.append(float(v.std()) if v.size > 1 else 0.0)
        ax.barh(ypos + (0.5 - k) * h, means, height=h, xerr=sds, color=color, ecolor=INK2, error_kw=dict(lw=0.9, capsize=2),
                label=f"{group} cells", zorder=3)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([b.replace("_", " ") + "\n" + "\n".join(textwrap.wrap(meaning_of(b), 34)) for b in band_names], fontsize=7, color=INK)
    ax.set_xlabel("attended pixels minus patch average, in patch standard deviations", fontsize=8, color=INK2)
    ax.grid(axis="x", color="#e6e5e1", zorder=0)
    ax.tick_params(colors=INK2, labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=f"{g} cells") for g, c in groups], fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle(heading or "What the CNN attends to, hot against cold cells", color=INK, fontsize=10, x=0.02, ha="left")
    foot = cartography.footer(fig, cfg, caption, (NOTE_HOT_COLD, NOTE_PROFILE)) if cfg is not None else 0.0
    fig.tight_layout(rect=(0, foot, 1, 0.95))
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path
