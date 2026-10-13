"""Plot functions. Each takes data + an output path and saves a PNG.

Style: one validated categorical order (blue, orange), thin marks, recessive grid, direct labels
instead of legends where there are only one or two series, ink colours for all text.
"""
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
DPI = 150


def _axes(figsize=(6.4, 4.2)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    return fig, ax


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_loss_curve(history: dict, out_path: str) -> None:
    """history = {'folds': [hist, ...], 'final': hist}; each hist has train_loss / val_loss lists."""
    fig, ax = _axes()
    folds = history.get("folds", [])
    for i, h in enumerate(folds):
        ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], color=BLUE, lw=1.4, alpha=0.55)
        if h["val_loss"]:
            ax.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], color=ORANGE, lw=1.4, alpha=0.55)
            if h.get("best_epoch"):
                ax.plot(h["best_epoch"], h["val_loss"][h["best_epoch"] - 1], "o", ms=5, color=ORANGE,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
    final = history.get("final")
    if final and final["train_loss"]:
        ax.plot(range(1, len(final["train_loss"]) + 1), final["train_loss"], color=INK, lw=1.6, ls="--")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (scaled units)")
    top = ax.get_ylim()[1]
    ax.text(0.98, 0.95, "train loss", color=BLUE, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.text(0.98, 0.88, "validation loss (held-out stations); dot = best epoch", color=ORANGE,
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    if final and final["train_loss"]:
        ax.text(0.98, 0.81, "dashed: final fit on all training stations", color=INK2,
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.set_ylim(top=top)
    _save(fig, out_path)


def plot_pred_vs_obs(y_true, y_pred, out_path: str, metrics: dict | None = None) -> None:
    fig, ax = _axes((5.2, 5.0))
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    hb = ax.hexbin(y_true, y_pred, gridsize=40, bins="log", mincnt=1, cmap="Blues", linewidths=0.0)
    lim = [0, max(float(y_true.max()), float(y_pred.max())) * 1.02]
    ax.plot(lim, lim, color=INK2, lw=1.0, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("observed PM2.5 (ug/m3)")
    ax.set_ylabel("predicted PM2.5 (ug/m3)")
    if metrics:
        ax.set_title(f"n = {len(y_true):,}   RMSE {metrics['rmse']:.2f}   r {metrics['pearson_r']:.2f}",
                     color=INK, fontsize=10, loc="left")
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("rows (log)", color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    _save(fig, out_path)


def plot_residual_hist(y_true, y_pred, out_path: str) -> None:
    fig, ax = _axes()
    res = np.asarray(y_pred).ravel() - np.asarray(y_true).ravel()
    ax.hist(res, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(0, color=INK2, lw=1.0)
    ax.axvline(res.mean(), color=ORANGE, lw=1.4)
    ax.text(res.mean(), ax.get_ylim()[1] * 0.97, f" mean {res.mean():+.2f}", color=ORANGE, fontsize=9, va="top")
    ax.set_xlabel("prediction - observation (ug/m3)")
    ax.set_ylabel("rows")
    _save(fig, out_path)


def plot_timeseries_sample(df, out_path: str, days: int = 7) -> None:
    """df: location_id, timestamp, y_true, y_pred (+ lo, hi). Shows the last `days` days of one test station."""
    station = df["location_id"].value_counts().idxmax()
    sub = df[df["location_id"] == station].sort_values("timestamp").tail(24 * days)
    fig, ax = _axes((8.0, 3.8))
    if {"lo", "hi"} <= set(sub.columns):
        ax.fill_between(sub["timestamp"], sub["lo"], sub["hi"].clip(upper=sub["y_true"].max() * 3),
                        color=BLUE, alpha=0.14, linewidth=0)
    ax.plot(sub["timestamp"], sub["y_true"], color=BLUE, lw=1.6)
    ax.plot(sub["timestamp"], sub["y_pred"], color=ORANGE, lw=1.6)
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_title(f"test station {station}", color=INK, fontsize=10, loc="left")
    ax.text(0.01, 0.95, "observed", color=BLUE, transform=ax.transAxes, va="top", fontsize=9)
    ax.text(0.01, 0.87, "predicted", color=ORANGE, transform=ax.transAxes, va="top", fontsize=9)
    if "lo" in sub.columns:
        ax.text(0.01, 0.79, "shaded: GeoCP interval", color=INK2, transform=ax.transAxes, va="top", fontsize=9)
    fig.autofmt_xdate()
    _save(fig, out_path)


def plot_geocp_coverage(per_station: list[dict], target: float, out_path: str) -> None:
    fig, ax = _axes((6.4, max(2.6, 0.5 * len(per_station) + 1.2)))
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ids = [str(r["location_id"]) for r in per_station]
    cov = [r["coverage"] for r in per_station]
    ypos = np.arange(len(ids))
    ax.barh(ypos, cov, color=BLUE, height=0.6)
    ax.axvline(target, color=ORANGE, lw=1.4)
    for y, c in zip(ypos, cov):
        ax.text(c + 0.01, y, f"{c:.0%}", va="center", color=INK, fontsize=9)
    ax.set_yticks(ypos, ids)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of readings inside the interval")
    ax.set_title(f"GeoCP coverage per test station (target {target:.0%}, orange line)", color=INK,
                 fontsize=10, loc="left")
    _save(fig, out_path)


def plot_search_convergence(trials: list[dict], out_path: str) -> None:
    """Validation loss of every Bayesian-search trial (dots) and the best so far (line)."""
    fig, ax = _axes()
    n = np.arange(1, len(trials) + 1)
    loss = np.array([t["loss"] for t in trials])
    ax.plot(n, loss, "o", ms=5, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.plot(n, np.minimum.accumulate(loss), color=ORANGE, lw=1.8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("trial")
    ax.set_ylabel("mean inner-fold validation loss")
    ax.text(0.98, 0.95, "trial", color=BLUE, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.text(0.98, 0.88, "best so far", color=ORANGE, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    _save(fig, out_path)


CLUSTER_STYLE = {                       # hot = under-prediction (orange), cold = over-prediction (blue)
    "High-High": ("#eb6834", 60, "o"), "Low-Low": ("#2a78d6", 60, "o"),
    "High-Low": ("#1baf7a", 45, "^"), "Low-High": ("#eda100", 45, "v"),
    "Not significant": ("#9a9992", 22, "o"),
}


def _cluster_scatter(ax, cells, edge=SURFACE):
    for name, (color, size, marker) in CLUSTER_STYLE.items():
        sub = cells[cells["cluster"] == name] if "cluster" in cells else cells.iloc[0:0]
        if len(sub):
            ax.scatter(sub["cx"], sub["cy"], s=size, c=color, marker=marker, edgecolors=edge, linewidths=0.8,
                       label=f"{name} ({len(sub)})", zorder=3)


def plot_lisa_map(cells, global_result: dict, out_path: str) -> None:
    """Grid cells coloured by LISA cluster (or by mean residual when LISA was not run)."""
    fig, ax = _axes((6.4, 5.6))
    ax.set_aspect("equal")
    if "cluster" in cells:
        _cluster_scatter(ax, cells)
        ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=INK2)
    else:
        sc = ax.scatter(cells["cx"], cells["cy"], c=cells["mean_residual"], cmap="RdBu_r", s=60,
                        edgecolors=SURFACE, vmin=-abs(cells["mean_residual"]).max(), vmax=abs(cells["mean_residual"]).max())
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label("mean residual (actual - predicted, ug/m3)", color=INK2)
    ax.ticklabel_format(useOffset=False, style="plain")       # full metres, no "1e6" offset over the title
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title(f"Global Moran's I = {global_result['I']:.3f}  (p = {global_result['p_sim']:.3f})",
                 color=INK, fontsize=10, loc="left")
    _save(fig, out_path)


def plot_lisa_on_bands(raster_path: str, cells, band_names: list[str], out_path: str, max_px: int = 700) -> None:
    """LISA clusters drawn over each morphology band of the Master GeoTIFF (one panel per band)."""
    import rasterio

    with rasterio.open(raster_path) as src:
        scale = max(1, int(np.ceil(max(src.width, src.height) / max_px)))
        data = src.read(out_shape=(src.count, max(1, src.height // scale), max(1, src.width // scale)))
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    n = data.shape[0]
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), facecolor=SURFACE, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        ax.set_facecolor(SURFACE)
        ax.imshow(data[i], extent=extent, origin="upper", cmap="Greys", interpolation="nearest")
        _cluster_scatter(ax, cells, edge="white")
        ax.set_title(band_names[i] if i < len(band_names) else f"band {i + 1}", color=INK, fontsize=10, loc="left")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(colors=INK2, labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), fontsize=8, frameon=False,
               labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=3.0, w_pad=2.0)
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
