"""Hot and cold cells of the high and the low end of each morphology band, over the whole NCR (--stage explain, next to the plain hot/cold).

A band's high end is the cells whose 320 m patch mean of that band, taken from the RASTER, is at or above an NCR percentile; its low end
is the cells at or below the low percentile. Only afterwards does the prediction come in: inside a group the HOT cells are the highest
predicted PM2.5 of the explained hour, the COLD cells the lowest, and Grad-CAM is run on those. Two levels are kept by default: a wide
`zone` (top and bottom 10 %) and an `extreme` end (top and bottom 0.1 %).

This describes what the model predicts and attends to where a feature is high or low. It does not say what would happen if the feature
were changed (no what-if is run).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from explain.gradcam import attention_profile, spread_selection

DEFAULTS = {
    "enabled": True,
    "levels": [{"name": "zone", "high_percentile": 90.0, "low_percentile": 10.0},
               {"name": "extreme", "high_percentile": 99.9, "low_percentile": 0.1}],
    "top_n": None,             # None = the explain.top_n / bottom_n / min_separation_m of the plain hot and cold cells
    "bottom_n": None,
    "min_separation_m": None,
    "feature_step_m": 50.0,    # the raster is averaged to cells of about this size to find each cell's patch mean
}
CANDIDATES = 20000             # as for the plain hot and cold cells: only the best candidates of each end are spread out
MIN_VALID_SHARE = 0.5          # a patch with less valid raster than this has no patch mean (the edge of the raster, water)


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **((cfg["output"].get("explain") or {}).get("feature_groups") or {})}


def patch_mean_grids(raster_path: str, patch_m: float, step_m: float = 50.0):
    """Per band, the mean of the raster over a patch_m x patch_m window around every cell of a ~step_m grid (NoData ignored; NaN where
    less than half of the window holds data). Returns (grids [C, h, w], transform, extent)."""
    from scipy.ndimage import uniform_filter

    from explain.morphology_sheet import read_bands

    data, transform, extent = read_bands(raster_path, step_m)
    window = max(1, int(round(patch_m / abs(transform.a))))
    valid = np.isfinite(data)
    size = (1, window, window)
    total = uniform_filter(np.where(valid, data, 0.0), size=size, mode="constant")
    share = uniform_filter(valid.astype("float64"), size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        grids = np.where(share >= MIN_VALID_SHARE, total / np.where(share > 0, share, 1.0), np.nan)
    return grids, transform, extent


def sample_grid(grid: np.ndarray, transform, xs, ys) -> np.ndarray:
    """The north-up `grid` value under each (x, y); NaN outside the grid."""
    col = np.floor((np.asarray(xs, dtype="float64") - transform.c) / transform.a).astype("int64")
    row = np.floor((np.asarray(ys, dtype="float64") - transform.f) / transform.e).astype("int64")
    ok = (row >= 0) & (row < grid.shape[0]) & (col >= 0) & (col < grid.shape[1])
    out = np.full(len(col), np.nan)
    out[ok] = grid[row[ok], col[ok]]
    return out


def thresholds(values: np.ndarray, high_pct: float, low_pct: float):
    """(low, high) band values at the two percentiles of the finite values, or None when the band cannot be split
    (no values, or the two limits coincide)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, [low_pct, high_pct])
    return (float(lo), float(hi)) if hi > lo else None


def membership(values: np.ndarray, low: float, high: float):
    """(in the high group, in the low group): values >= high, values <= low (NaN is in neither). Many cells can tie at a limit (e.g.
    a band that is 0 over water), so a group can be larger than its percentile says; its true size is reported."""
    with np.errstate(invalid="ignore"):
        return values >= high, values <= low


def pick_hot_cold(pred: np.ndarray, xs, ys, member: np.ndarray, n_hot: int, n_cold: int, min_separation_m: float):
    """Positions (into pred / xs / ys) of the n_hot highest-predicted and n_cold lowest-predicted cells of `member`, no two of the same
    kind closer than min_separation_m. A cell is never both: when the group is small, the hot picks keep it."""
    idx = np.flatnonzero(member)
    order = idx[np.argsort(pred[idx], kind="stable")]
    xs, ys = np.asarray(xs), np.asarray(ys)

    def pick(best_first, n):
        flat = best_first[:CANDIDATES]
        return flat[spread_selection(xs[flat], ys[flat], n, float(min_separation_m))] if n > 0 and len(flat) else np.empty(0, int)

    hot = pick(order[::-1], n_hot)
    return hot, pick(order[~np.isin(order, hot)], n_cold)


def city_index(transform, shape, boundary):
    """(grid of city numbers 1..n, 0 outside, city names): which NCR city each cell of the grid belongs to."""
    from rasterio.features import rasterize

    names = [str(c) for c in boundary["city"]]
    shapes = [(g, i + 1) for i, g in enumerate(boundary.geometry)]
    return rasterize(shapes, out_shape=tuple(shape), transform=transform, fill=0, all_touched=True, dtype="int32"), names


def city_table(pred: np.ndarray, city_of_cell: np.ndarray, city_names: list[str], memberships: dict) -> pd.DataFrame:
    """Per city, band and level: its cells, the share of them in the high and in the low group, and the mean prediction in each.
    `memberships`: {(band, level): (high mask, low mask)} over the same cells as `pred`; `city_of_cell`: 0 = outside every city."""
    rows = []
    for number, city in enumerate(city_names, start=1):
        inside = city_of_cell == number
        n = int(inside.sum())
        if n == 0:
            continue
        for (band, level), (high, low) in memberships.items():
            row = {"city": city, "band": band, "level": level, "cells": n, "mean_prediction": float(pred[inside].mean())}
            for side, mask in (("high", high), ("low", low)):
                m = inside & mask
                row[f"share_{side}"] = float(m.sum() / n)
                row[f"mean_prediction_{side}"] = float(pred[m].mean()) if m.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


NOTE_GROUPS = ("A cell is in the HIGH group of a band when the raster value of its 320 m patch is at or above the NCR percentile named in "
               "the heading, in the LOW group when at or below the low percentile. Inside a group, HOT cells are the highest predicted "
               "PM2.5 of the explained hour, COLD the lowest, no two closer than the separation set in the config.")
NOTE_ASSOCIATION = ("This shows what the model predicts and where its CNN looks where a feature is high or low. It does not show what "
                    "would happen if the feature were changed.")


def run(cfg: dict, *, pred_map, transform, ok, names, raster_path, explain_cells, out_dir, ts, caption, stations=None,
        dpi=150, px: int, res: float) -> dict:
    """The whole pass. `ok`: the cells of pred_map that may be picked (finite, inside the NCR). `explain_cells(xs, ys)` ->
    (raw patches, cam, predictions, boolean mask of the cells kept): Grad-CAM for those cells. Writes into out_dir/feature_groups/ and
    returns the summary dictionary."""
    from rasterio.transform import xy as cell_xy_of

    from explain.feature_groups_plots import plot_group_comparison, plot_group_map
    from explain.gradcam_plots import plot_cam_overlays
    from output import cartography

    s, ex = settings(cfg), cfg["output"]["explain"]
    n_hot = int(s["top_n"] if s["top_n"] is not None else ex["top_n"])
    n_cold = int(s["bottom_n"] if s["bottom_n"] is not None else ex["bottom_n"])
    sep = float(s["min_separation_m"] if s["min_separation_m"] is not None else ex.get("min_separation_m", 0))
    folder = Path(out_dir) / "feature_groups"
    folder.mkdir(parents=True, exist_ok=True)

    grids, f_transform, f_extent = patch_mean_grids(raster_path, px * res, float(s["feature_step_m"]))
    bands = list(names[:grids.shape[0]])
    r, c = np.nonzero(ok)
    xs, ys = (np.asarray(v, dtype="float64") for v in cell_xy_of(transform, r, c, offset="center"))
    pred = pred_map[r, c].astype("float64")

    boundary = cartography.active_boundary(cfg)
    city_of_cell, city_names = None, []
    if boundary is not None:
        grid_of_cities, city_names = city_index(transform, pred_map.shape, boundary)
        city_of_cell = grid_of_cities[r, c]

    picks, attention, memberships, summary_rows, skipped = [], [], {}, [], []
    for b, band in enumerate(bands):
        values = sample_grid(grids[b], f_transform, xs, ys)
        for level in s["levels"]:
            name = str(level["name"])
            limits = thresholds(values, float(level["high_percentile"]), float(level["low_percentile"]))
            if limits is None:
                skipped.append({"band": band, "level": name, "reason": "the band is constant (or empty) inside the NCR"})
                continue
            low, high = limits
            high_m, low_m = membership(values, low, high)
            memberships[(band, name)] = (high_m, low_m)
            for side, mask, limit in (("high", high_m, high), ("low", low_m, low)):
                hot_i, cold_i = pick_hot_cold(pred, xs, ys, mask, n_hot, n_cold, sep)
                chosen = np.concatenate([hot_i, cold_i]).astype(int)
                row = {"band": band, "level": name, "side": side, "limit": limit, "cells": int(mask.sum()),
                       "mean_prediction": float(pred[mask].mean()) if mask.any() else np.nan,
                       "sd_prediction": float(pred[mask].std()) if mask.any() else np.nan,
                       "hot_requested": n_hot, "cold_requested": n_cold, "hot_picked": 0, "cold_picked": 0}
                if len(chosen):
                    raw, cam, p_now, keep = explain_cells(xs[chosen], ys[chosen])
                    kx, ky = xs[chosen][keep], ys[chosen][keep]
                    kind = np.where(np.flatnonzero(keep) < len(hot_i), "hot", "cold")
                    number = np.zeros(len(kind), dtype=int)
                    for k in ("hot", "cold"):
                        number[kind == k] = np.arange(1, int((kind == k).sum()) + 1)
                    row["hot_picked"], row["cold_picked"] = int((kind == "hot").sum()), int((kind == "cold").sum())
                    prof = attention_profile(raw, cam, list(names), ex["attention_quantile"])
                    for p in prof:
                        i = p["patch"]
                        attention.append({"group_band": band, "level": name, "side": side, "kind": kind[i], "cell_number": int(number[i]),
                                          "x": float(kx[i]), "y": float(ky[i]), "predicted": float(p_now[i]),
                                          **{k: v for k, v in p.items() if k != "patch"}})
                    for i in range(len(kx)):
                        picks.append({"group_band": band, "level": name, "side": side, "kind": kind[i], "number": int(number[i]),
                                      "x": float(kx[i]), "y": float(ky[i]), "predicted": float(p_now[i])})
                    order = np.argsort(kind == "cold", kind="stable")             # hot rows first, then cold
                    titles = [f"{side.upper()} {band.replace('_', ' ')}\n{kind[i].upper()} #{number[i]}\n{p_now[i]:.1f} µg/m³\n"
                              f"({kx[i]:.0f}, {ky[i]:.0f})" for i in order]
                    plot_cam_overlays(raw[order], cam[order], names, titles, str(folder / f"gradcam_{band}_{name}_{side}.png"),
                                      ex["attention_quantile"], dpi, cfg=cfg, px_m=res, caption=caption, notes=(NOTE_GROUPS, NOTE_ASSOCIATION),
                                      heading=(f"Grad-CAM, {side} {band.replace('_', ' ')} ({name}: {'>=' if side == 'high' else '<='} "
                                               f"{limit:.3g}), {ts:%Y-%m-%d %H:%M}"))
                summary_rows.append(row)
            plot_group_map(grids[b], f_extent, f_transform, band, name, (low, high), [p for p in picks if p["group_band"] == band and p["level"] == name],
                           cfg, str(folder / f"map_{band}_{name}.png"), stations=stations, caption=caption, dpi=dpi,
                           heading=f"{band.replace('_', ' ')}: where it is high and low in the NCR, and the hot and cold cells of each end "
                                   f"({name}, {ts:%Y-%m-%d %H:%M})")

    table = pd.DataFrame(summary_rows)
    table.to_csv(folder / "feature_group_summary.csv", index=False)
    pd.DataFrame(picks).to_csv(folder / "feature_group_cells.csv", index=False)
    pd.DataFrame(attention).to_csv(folder / "feature_group_attention.csv", index=False)
    if city_of_cell is not None and memberships:
        city_table(pred, city_of_cell, city_names, memberships).to_csv(folder / "feature_group_cities.csv", index=False)
    if len(table):
        plot_group_comparison(table, float(pred.mean()), str(folder / "feature_group_comparison.png"), cfg=cfg, caption=caption, dpi=dpi,
                              heading=f"Predicted PM2.5 where each band is high and where it is low ({ts:%Y-%m-%d %H:%M})")
    return {"folder": str(folder), "levels": s["levels"], "hot_per_group": n_hot, "cold_per_group": n_cold, "min_separation_m": sep,
            "cells_ranked": int(len(pred)), "ncr_mean_prediction": float(pred.mean()) if len(pred) else None,
            "groups": table.to_dict("records") if len(table) else [], "skipped": skipped,
            "reading": "association only: what the model predicts and attends to where a band is high or low, not a what-if effect"}
