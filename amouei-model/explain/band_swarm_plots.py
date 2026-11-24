"""The Grad-CAM band scores drawn the way the XGBoost branch draws its SHAP summary plot (xgboost/reporting/figures.py, shap.summary_plot):
the same figure size, fonts, dotted row lines, red-blue colours and colourbar, so the two figures can sit side by side in a paper. The layout
code below follows shap's summary_plot (beeswarm layout, colours between the 5th and 95th percentile, Low/High colourbar); the shap package
itself is not needed. See explain/band_swarm.py for what the score means. The explanation of the figure belongs in the caption, not on it.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RED_BLUE = LinearSegmentedColormap.from_list("shap_red_blue", ["#008bfb", "#ff0051"])      # shap.plots.colors.red_blue
BAR_BLUE = "#008bfb"
AXIS = "#333333"
FIG_SIZE = (7.4, 5.0)                    # the XGBoost beeswarm: figsize=(7.4, 5.0), saved at dpi 140
DPI = 140


def swarm_offsets(x: np.ndarray, row_height: float = 0.4, nbins: int = 100, seed: int = 0) -> np.ndarray:
    """shap's beeswarm layout: dots are binned along x and stacked alternately above and below the row line inside their bin, scaled so the
    widest stack spans 0.9 * row_height each way."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    ys = np.zeros(n)
    if n == 0:
        return ys
    quant = np.round(nbins * (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8))
    inds = np.argsort(quant + np.random.default_rng(seed).standard_normal(n) * 1e-6)
    layer, last_bin = 0, -1
    for ind in inds:
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]
    return ys * (0.9 * (row_height / np.max(ys + 1)))


def plot_band_swarm(scores: np.ndarray, levels: np.ndarray, bands: list[str], path: str, dpi: int = DPI, *, cfg: dict | None = None,
                    caption: str | None = None, heading: str | None = None, n_cells: int | None = None, xlabel: str | None = None,
                    order=None) -> str:
    """`scores` and `levels` [cells, bands]: the signed attention score (or another signed effect, with its own `xlabel`) and the 0..1
    colour level of the band's patch mean. `order` = the band indices from the top row down (default: the largest mean |score| first).
    `cfg` and `caption` are accepted for symmetry with the other figures; like the SHAP plot, this one carries no notes."""
    order = np.argsort(-np.nanmean(np.abs(scores), axis=0), kind="stable") if order is None else np.asarray(order)
    rows = order[::-1]                                                              # shap draws the top row last (highest y)
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()
    for pos, j in enumerate(rows):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        ok = np.isfinite(scores[:, j])
        x, c = scores[ok, j], levels[ok, j]
        y = pos + swarm_offsets(x)
        gray = ~np.isfinite(c)
        if gray.any():
            ax.scatter(x[gray], y[gray], color="#777777", s=16, linewidth=0, zorder=3, rasterized=len(x) > 500)
        ax.scatter(x[~gray], y[~gray], cmap=RED_BLUE, vmin=0, vmax=1, s=16, c=c[~gray], linewidth=0, zorder=3, rasterized=len(x) > 500)
    mappable = cm.ScalarMappable(cmap=RED_BLUE)
    mappable.set_array([0, 1])
    cb = plt.colorbar(mappable, ax=ax, ticks=[0, 1], aspect=80)
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Band value in the patch", size=12, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)
    ax.axvline(x=0, color="#999999", zorder=-1)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    for side in ("right", "top", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(color=AXIS, labelcolor=AXIS)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([bands[j] for j in rows], fontsize=13)
    ax.tick_params("y", length=20, width=0.5, which="major", color="none")
    ax.tick_params("x", labelsize=11)
    ax.set_ylim(-1, len(rows))
    ax.set_xlabel(xlabel or "Grad-CAM attention score", fontsize=13)
    ax.set_title(heading or f"Grad-CAM attention score{f' — {n_cells} NCR cells' if n_cells else ''}", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_band_importance(imp, bands: list[str], order, path: str, dpi: int = DPI, *, cfg: dict | None = None, caption: str | None = None,
                         heading: str | None = None, xlabel: str | None = None) -> str:
    """The bar version in shap's bar-plot style: mean |score| per band, largest on top (`imp`: importance() output, a band column; the thin
    whiskers are the 95 % bootstrap interval of that mean)."""
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()
    rows = imp.iloc[list(order)].iloc[::-1]
    ypos = np.arange(len(rows))
    mean = rows["mean_abs_score"].to_numpy()
    err = np.clip(np.vstack([mean - rows["ci_low"].to_numpy(), rows["ci_high"].to_numpy() - mean]), 0, None)
    ax.barh(ypos, mean, 0.7, align="center", color=BAR_BLUE, xerr=err, ecolor="#555555", error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(ypos)
    ax.set_yticklabels(list(rows["band"]), fontsize=13)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    for side in ("right", "top", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params("x", labelsize=11)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlabel(xlabel or "mean(|Grad-CAM attention score|) (patch std)", fontsize=13)
    ax.set_title(heading or "Mean Grad-CAM attention per band", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
