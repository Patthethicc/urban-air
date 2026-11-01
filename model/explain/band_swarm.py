"""The Grad-CAM counterpart of a SHAP summary plot (--stage explain): one dot per NCR cell, one row per morphology band.

For a sample of cells across the NCR (not only the hot and cold ones), Grad-CAM is run on each cell's 320 m patch and, per band, the
attention score of gradcam.attention_profile is kept: (band on the attended pixels - patch average) / patch standard deviation. A dot's x
is that signed score and its colour is the band's value in the patch (patch mean, low to high), exactly the way a SHAP beeswarm shows
'impact' against 'feature value'. Rows are ordered by mean |score|, the largest on top.

It describes where the CNN looks and what lies under it, relative to each patch. It is not an effect size in ug/m3 like a SHAP value, and it
is not a cause.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from explain.gradcam import attention_profile

DEFAULTS = {"enabled": True, "n_cells": 300, "seed": 0, "chunk": 128}
COLOUR_CLIP = (5.0, 95.0)       # the colour scale runs between these percentiles of the band's patch means (as the SHAP plot does)


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **((cfg["output"].get("explain") or {}).get("band_swarm") or {})}


def sample_cells(ok: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Flat indices of up to `n` cells of the boolean grid `ok`, drawn without replacement (same seed, same cells), in grid order."""
    flat = np.flatnonzero(ok.ravel())
    if len(flat) <= n:
        return flat
    return np.sort(np.random.default_rng(seed).choice(flat, size=n, replace=False))


def attention_scores(raw: np.ndarray, cam: np.ndarray, band_names: list[str], quantile: float):
    """(scores [cells, bands], patch means [cells, bands]) from attention_profile; NaN where a cell has no attended pixels."""
    n, b = len(cam), raw.shape[1]
    scores, values = np.full((n, b), np.nan), np.full((n, b), np.nan)
    index = {name: j for j, name in enumerate(band_names[:b])}
    for r in attention_profile(raw, cam, list(band_names), quantile):
        scores[r["patch"], index[r["band"]]] = r["std_units"]
        values[r["patch"], index[r["band"]]] = r["patch_mean"]
    return scores, values


def row_order(scores: np.ndarray) -> np.ndarray:
    """Band indices from the largest mean |score| to the smallest (the top row of a SHAP plot is the most important)."""
    return np.argsort(-np.nanmean(np.abs(scores), axis=0), kind="stable")


def colour_levels(values: np.ndarray, clip=COLOUR_CLIP) -> np.ndarray:
    """Each value scaled to 0..1 between the clip percentiles of its own column (NaN stays NaN; a constant column is 0.5)."""
    out = np.full(values.shape, np.nan)
    for j in range(values.shape[1]):
        col = values[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        lo, hi = np.percentile(col[ok], clip)
        out[ok, j] = 0.5 if hi <= lo else np.clip((col[ok] - lo) / (hi - lo), 0.0, 1.0)
    return out


def importance(scores: np.ndarray, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per band: mean score, mean |score|, and a 95 % bootstrap interval of the mean |score| over cells."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(scores.shape[1]):
        s = scores[np.isfinite(scores[:, j]), j]
        if s.size == 0:
            rows.append({"mean_score": np.nan, "mean_abs_score": np.nan, "ci_low": np.nan, "ci_high": np.nan, "cells": 0})
            continue
        a = np.abs(s)
        boot = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"mean_score": float(s.mean()), "mean_abs_score": float(a.mean()), "ci_low": float(lo), "ci_high": float(hi),
                     "cells": int(s.size)})
    return pd.DataFrame(rows)


def run(cfg: dict, *, pred_map, transform, ok, names, explain_cells, out_dir, ts, caption, dpi=150) -> dict:
    """Sample cells, run Grad-CAM on them in chunks, write gradcam_band_swarm.png, gradcam_band_importance.png and gradcam_band_swarm.csv
    into out_dir, and return the summary. `explain_cells(xs, ys)` -> (raw patches, cam, predictions, mask of the cells kept)."""
    from rasterio.transform import xy as cell_xy_of

    from explain.band_swarm_plots import plot_band_importance, plot_band_swarm

    s, ex = settings(cfg), cfg["output"]["explain"]
    flat = sample_cells(ok, int(s["n_cells"]), int(s["seed"]))
    r, c = np.unravel_index(flat, ok.shape)
    xs, ys = (np.asarray(v, dtype="float64") for v in cell_xy_of(transform, r, c, offset="center"))
    scores, values, preds, kx, ky = [], [], [], [], []
    for lo in range(0, len(flat), int(s["chunk"])):
        sl = slice(lo, lo + int(s["chunk"]))
        raw, cam, pred, keep = explain_cells(xs[sl], ys[sl])
        if len(cam) == 0:
            continue
        sc, va = attention_scores(raw, cam, list(names), ex["attention_quantile"])
        scores.append(sc), values.append(va), preds.append(pred)
        kx.append(xs[sl][keep]), ky.append(ys[sl][keep])
    if not scores:
        return {"skipped": "no cell of the sample has a full history, so there is nothing to explain"}
    scores, values = np.vstack(scores), np.vstack(values)
    preds, kx, ky = np.concatenate(preds), np.concatenate(kx), np.concatenate(ky)
    bands = list(names[:scores.shape[1]])

    folder = Path(out_dir)
    table = pd.DataFrame({"x": kx, "y": ky, "predicted": preds})
    for j, band in enumerate(bands):
        table[f"score_{band}"], table[f"patch_mean_{band}"] = scores[:, j], values[:, j]
    table.to_csv(folder / "gradcam_band_swarm.csv", index=False)
    imp = importance(scores, seed=int(s["seed"]))
    imp.insert(0, "band", bands)
    order = row_order(scores)
    heading = f"Grad-CAM attention score — {len(table)} NCR cells, {ts:%Y-%m-%d %H:%M}"
    plot_band_swarm(scores, colour_levels(values), bands, str(folder / "gradcam_band_swarm.png"), dpi, cfg=cfg, caption=caption,
                    heading=heading, n_cells=len(table))
    plot_band_importance(imp, bands, order, str(folder / "gradcam_band_importance.png"), dpi, cfg=cfg, caption=caption,
                         heading=f"Mean Grad-CAM attention — {len(table)} NCR cells, {ts:%Y-%m-%d %H:%M}")
    return {"cells_sampled": int(len(flat)), "cells_explained": int(len(table)), "seed": int(s["seed"]),
            "order": [bands[i] for i in order], "bands": imp.to_dict("records"),
            "reading": "x = attended pixels minus patch average in patch std units (positive: the CNN looked at the higher values of the band); "
                       "colour = the band's patch mean; where the CNN looks, not an effect in ug/m3, not a cause"}
