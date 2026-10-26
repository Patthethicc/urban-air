"""Grad-CAM on the city map: the predicted PM2.5 surface of the explained hour with the hot and cold cells that were explained,
numbered like the rows of gradcam_hot_cells.png / gradcam_cold_cells.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explain.gradcam_plots import COLD, HOT, NOTE_HOT_COLD  # noqa: E402
from output import aqi_maps, cartography  # noqa: E402
from reporting.plotters import INK2, SURFACE  # noqa: E402

NOTE_MAP = ("Colours: the predicted PM2.5 (µg/m³) of this hour, as on the prediction maps. Each marker is one explained cell; its number "
            "is the row of the close-up figure that shows which pixels of its 320 m patch drove its prediction.")


def plot_gradcam_map(pred_map: np.ndarray, transform, crs, cells: list[dict], cfg: dict, path: str, title: str, *,
                     stations=None, caption: str | None = None, dpi: int = 150) -> str:
    """`cells`: dicts with group ('hot' or 'cold'), number, x, y (map coordinates) and predicted."""
    fig, ax = plt.subplots(figsize=(6.8, 8.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    image = aqi_maps.draw_surface(ax, pred_map, transform, crs, cfg, None, aqi_maps._stations(cfg, stations), title, style="continuous")
    for group, color, marker in (("hot", HOT, "o"), ("cold", COLD, "s")):
        pts = [c for c in cells if c["group"] == group]
        if not pts:
            continue
        ax.scatter([c["x"] for c in pts], [c["y"] for c in pts], s=120, marker=marker, facecolor=color, edgecolor="white",
                   linewidth=1.4, zorder=9)
        for c in pts:
            ax.text(c["x"], c["y"], str(c["number"]), fontsize=6.5, color="white", ha="center", va="center", zorder=10,
                    fontweight="bold", path_effects=[pe.withStroke(linewidth=0.6, foreground=color)])
    fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02).set_label(aqi_maps.UNIT, color=INK2, fontsize=8)
    n_hot, n_cold = sum(c["group"] == "hot" for c in cells), sum(c["group"] == "cold" for c in cells)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=HOT, markeredgecolor="white", markersize=9,
                      label=f"hot: the {n_hot} highest predicted cells"),
               Line2D([0], [0], marker="s", color="none", markerfacecolor=COLD, markeredgecolor="white", markersize=9,
                      label=f"cold: the {n_cold} lowest predicted cells"), cartography.no_data_patch()]
    foot = cartography.footer(fig, cfg, caption, (NOTE_MAP, NOTE_HOT_COLD))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, foot + 0.005), ncol=3, fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.tight_layout(rect=(0, foot + 0.05, 1, 1))
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path
