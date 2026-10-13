"""Grad-CAM overlays on the input morphology patches (Output.txt)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from reporting.plotters import INK, INK2, ORANGE, SURFACE  # noqa: E402


def plot_cam_overlays(raw_patches: np.ndarray, cam: np.ndarray, band_names: list[str], titles: list[str],
                      path: str, quantile: float = 0.8, dpi: int = 150) -> str:
    """One row per cell: every morphology band (grey) with the attended region outlined in orange, and the
    Grad-CAM map itself in the last column."""
    n, bands = len(cam), raw_patches.shape[1]
    fig, axes = plt.subplots(n, bands + 1, figsize=(2.3 * (bands + 1), 2.3 * n + 0.6), facecolor=SURFACE, squeeze=False)
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
                if i == 0:
                    ax.set_title(band_names[b] if b < len(band_names) else f"band {b + 1}", color=INK, fontsize=8)
            else:
                ax.imshow(cam[i], cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
                if i == 0:
                    ax.set_title("Grad-CAM", color=INK, fontsize=8)
        axes[i, 0].set_ylabel(titles[i], color=INK2, fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path
