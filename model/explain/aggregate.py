"""Grad-CAM over several hours of one day, pooled (--stage explain with several snapshot hours, output.explain.aggregate).

Every hour is explained on its own first (explain/<day>_<hhmm>/). This module then reads the tables those runs wrote and draws the same
figures once more over all hours together, plus one chart that puts the hours side by side:

    gradcam_band_profile.png      hot against cold cells, every hour's hot and cold cells pooled
    gradcam_band_swarm.png        the SHAP-style swarm, the sampled NCR cells of every hour pooled
    gradcam_band_importance.png   its bar version
    band_ablation.png / _swarm.png  the impact counterpart, pooled
    gradcam_band_by_hour.png      mean |attention| per band, one bar per hour and one for all hours
    *.csv                         the pooled tables

Nothing is recomputed here: the cells are the ones each hour already explained, so the pooled figure is exactly the four hours put together.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from explain import band_ablation, band_swarm  # noqa: E402
from explain.band_swarm_plots import AXIS, FIG_SIZE, plot_band_importance, plot_band_swarm  # noqa: E402

HOUR_COLOURS = ["#6baed6", "#3182bd", "#fd8d3c", "#e6550d", "#9e9ac8", "#756bb1"]      # one per hour, cycled
ALL_COLOUR = "#333333"


def read_hours(folders: dict, filename: str) -> pd.DataFrame:
    """The table `filename` of every hour's folder, stacked, with an `hour` column (hours that lack the file are left out)."""
    parts = []
    for hour, folder in folders.items():
        path = Path(folder) / filename
        if path.is_file():
            parts.append(pd.read_csv(path).assign(hour=hour))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def bands_of(table: pd.DataFrame, prefix: str) -> list[str]:
    return [c[len(prefix):] for c in table.columns if c.startswith(prefix) and not c.startswith(prefix + "abs_")]


def by_hour_table(swarm: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    """Mean |attention score| per band for each hour and for all hours together (rows = bands)."""
    out = {"band": bands}
    for hour, part in swarm.groupby("hour", sort=True):
        out[hour] = [float(np.nanmean(np.abs(part[f"score_{b}"]))) for b in bands]
    out["all hours"] = [float(np.nanmean(np.abs(swarm[f"score_{b}"]))) for b in bands]
    return pd.DataFrame(out)


def plot_band_by_hour(table: pd.DataFrame, path: str, dpi: int = 140, heading: str | None = None) -> str:
    """Grouped bars: for each band (largest overall on top) one bar per hour and a dark bar for all hours together."""
    cols = [c for c in table.columns if c not in ("band", "all hours")]
    order = np.argsort(-table["all hours"].to_numpy(), kind="stable")
    rows = table.iloc[order].iloc[::-1].reset_index(drop=True)
    n = len(cols) + 1
    h = 0.8 / n
    fig, ax = plt.subplots(figsize=(FIG_SIZE[0], 0.9 * len(rows) + 1.8))
    ypos = np.arange(len(rows))
    for k, col in enumerate(cols + ["all hours"]):
        colour = ALL_COLOUR if col == "all hours" else HOUR_COLOURS[k % len(HOUR_COLOURS)]
        ax.barh(ypos + (k - (n - 1) / 2) * h, rows[col], height=h, color=colour, label=col, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(list(rows["band"]), fontsize=12)
    ax.set_xlabel("mean(|Grad-CAM attention score|) (patch std)", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    for side in ("right", "top", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(color=AXIS, labelcolor=AXIS)
    ax.set_title(heading or "Mean Grad-CAM attention per band, by hour", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def run(cfg: dict, *, folders: dict, out_dir, names: list[str], caption: str | None, dpi: int, date: str) -> dict:
    """Pool the hours explained in `folders` ({'08:00': folder, ...}) into `out_dir`. Returns the summary dict."""
    from explain.gradcam_plots import plot_band_profile

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hours = list(folders)
    label = f"{date}, all hours ({', '.join(hours)})"
    summary: dict = {"hours": hours}

    attention = read_hours(folders, "gradcam_attention.csv")
    if len(attention):
        attention.to_csv(out / "gradcam_attention.csv", index=False)
        plot_band_profile(attention.to_dict("records"), list(names), str(out / "gradcam_band_profile.png"), dpi, cfg=cfg, caption=caption,
                          heading=f"What the CNN attends to, hot against cold cells ({label})")
        summary["profile_cells"] = int(len(attention))

    swarm = read_hours(folders, "gradcam_band_swarm.csv")
    if len(swarm):
        bands = bands_of(swarm, "score_")
        scores = swarm[[f"score_{b}" for b in bands]].to_numpy(dtype=float)
        values = swarm[[f"patch_mean_{b}" for b in bands]].to_numpy(dtype=float)
        swarm.to_csv(out / "gradcam_band_swarm.csv", index=False)
        imp = band_swarm.importance(scores, seed=int(band_swarm.settings(cfg)["seed"]))
        imp.insert(0, "band", bands)
        order = band_swarm.row_order(scores)
        plot_band_swarm(scores, band_swarm.colour_levels(values), bands, str(out / "gradcam_band_swarm.png"), dpi, cfg=cfg, caption=caption,
                        heading=f"Grad-CAM attention score — {len(swarm)} cell-hours, {label}", n_cells=len(swarm))
        plot_band_importance(imp, bands, order, str(out / "gradcam_band_importance.png"), dpi, cfg=cfg, caption=caption,
                             heading=f"Mean Grad-CAM attention — {len(swarm)} cell-hours, {label}")
        by_hour = by_hour_table(swarm, bands)
        by_hour.to_csv(out / "gradcam_band_by_hour.csv", index=False)
        plot_band_by_hour(by_hour, str(out / "gradcam_band_by_hour.png"), heading=f"Mean Grad-CAM attention per band, by hour ({date})")
        summary["band_swarm"] = {"cell_hours": int(len(swarm)), "order": [bands[i] for i in order], "bands": imp.to_dict("records"),
                                 "by_hour": by_hour.to_dict("records")}

    ablation = read_hours(folders, "band_ablation.csv")
    if len(ablation):
        bands = bands_of(ablation, "delta_")
        signed = ablation[[f"delta_{b}" for b in bands]].to_numpy(dtype=float)
        absolute = ablation[[f"abs_delta_{b}" for b in bands]].to_numpy(dtype=float)
        patch_mean = ablation[[f"patch_mean_{b}" for b in bands]].to_numpy(dtype=float)
        ablation.to_csv(out / "band_ablation.csv", index=False)
        seed = int(band_ablation.settings(cfg)["seed"])
        imp = band_ablation.importance(absolute, signed, seed=seed)
        imp.insert(0, "band", bands)
        order = np.argsort(-imp["mean_abs_score"].to_numpy(), kind="stable")
        plot_band_importance(imp, bands, order, str(out / "band_ablation.png"), dpi,
                             heading=f"Band impact — {len(ablation)} cell-hours, {label}", xlabel="mean(|change in prediction|) (µg/m³)")
        plot_band_swarm(signed, band_swarm.colour_levels(patch_mean), bands, str(out / "band_ablation_swarm.png"), dpi,
                        heading=f"Prediction change — {len(ablation)} cell-hours, {label}",
                        xlabel="change in the prediction when the band is swapped (µg/m³)", order=order)
        summary["band_ablation"] = {"cell_hours": int(len(ablation)), "order": [bands[i] for i in order], "bands": imp.to_dict("records")}
    return summary
