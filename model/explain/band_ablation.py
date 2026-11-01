"""How much each morphology band moves the prediction (--stage explain): the impact counterpart of the Grad-CAM attention figures.

For the same sample of NCR cells as the band swarm, one band at a time is replaced by that band of ANOTHER cell (a permutation across the
sample, so the band keeps a realistic distribution) while the cell's other bands and its PM2.5 history stay as they are, and the model predicts
again. The change in the prediction, in ug/m3, is the band's effect for that cell:

    delta = prediction(band swapped) - prediction(as it is)

Mean |delta| over cells is the band's impact (the same kind of number as the mean |SHAP| bar chart, in the model's own units); the signed
delta per cell, coloured by the band's value in the patch, is the beeswarm. The swap is repeated `repeats` times with different partners and
the effects are averaged, so one lucky partner does not decide. This is a model-reliance test, not a statement about the real world, and the
bands are correlated (a swapped band can also look odd next to the unswapped ones), so read it as the model's sensitivity to each band.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from explain import band_swarm

DEFAULTS = {"enabled": True, "n_cells": None, "repeats": 5, "seed": 0, "chunk": 256}      # n_cells None = the same as band_swarm.n_cells


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **((cfg["output"].get("explain") or {}).get("band_ablation") or {})}


def swap_partner(n: int, rng) -> np.ndarray:
    """For each of n cells the index of the OTHER cell whose band it receives: one random cycle, so nobody keeps their own (n > 1)."""
    order = rng.permutation(n)
    partner = np.empty(n, dtype=int)
    partner[order] = np.roll(order, 1)
    return partner


def permute_band(raw: np.ndarray, band: int, partner: np.ndarray) -> np.ndarray:
    """A copy of the patches [cells, bands, h, w] in which `band` of every cell is the band of its partner; all else is untouched."""
    out = raw.copy()
    out[:, band] = raw[partner, band]
    return out


def band_effects(predict, raw: np.ndarray, feats: np.ndarray, repeats: int, seed: int = 0, chunk: int = 256):
    """(baseline [cells], signed [cells, bands], absolute [cells, bands]) with `predict(raw, feats)` -> ug/m3 per cell. signed = the mean
    over the repeats of the change; absolute = the mean over the repeats of its size (so opposite changes do not cancel)."""
    def forward(patches):
        return np.concatenate([predict(patches[i:i + chunk], feats[i:i + chunk]) for i in range(0, len(patches), chunk)])

    n, bands = len(raw), raw.shape[1]
    base = forward(raw)
    signed, absolute = np.zeros((n, bands)), np.zeros((n, bands))
    if n < 2:
        return base, signed, absolute
    rng = np.random.default_rng(seed)
    for b in range(bands):
        for _ in range(repeats):
            delta = forward(permute_band(raw, b, swap_partner(n, rng))) - base
            signed[:, b] += delta / repeats
            absolute[:, b] += np.abs(delta) / repeats
    return base, signed, absolute


def importance(absolute: np.ndarray, signed: np.ndarray, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per band: mean |change| over cells with a 95 % bootstrap interval, and the mean signed change."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(absolute.shape[1]):
        a = absolute[:, j]
        boot = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1) if a.size else np.array([np.nan, np.nan])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"mean_abs_score": float(a.mean()) if a.size else np.nan, "ci_low": float(lo), "ci_high": float(hi),
                     "mean_signed": float(signed[:, j].mean()) if a.size else np.nan, "cells": int(a.size)})
    return pd.DataFrame(rows)


def run(cfg: dict, *, ok, transform, names, cell_inputs, predict, out_dir, ts, dpi=150) -> dict:
    """Sample the cells (the same ones as the band swarm), swap each band in turn and write band_ablation.png (bars), band_ablation_swarm.png
    (one dot per cell), band_ablation.csv into out_dir. `cell_inputs(xs, ys)` -> (raw patches, temporal features, kept mask);
    `predict(raw, feats)` -> ug/m3."""
    from rasterio.transform import xy as cell_xy_of

    from explain.band_swarm_plots import plot_band_importance, plot_band_swarm

    s, sw = settings(cfg), band_swarm.settings(cfg)
    n_cells = int(s["n_cells"] if s["n_cells"] is not None else sw["n_cells"])
    flat = band_swarm.sample_cells(ok, n_cells, int(sw["seed"]))
    r, c = np.unravel_index(flat, ok.shape)
    xs, ys = (np.asarray(v, dtype="float64") for v in cell_xy_of(transform, r, c, offset="center"))
    raw, feats, keep = cell_inputs(xs, ys)
    if len(raw) < 2:
        return {"skipped": "fewer than two cells of the sample have a full history, so there is nothing to swap"}
    bands = list(names[:raw.shape[1]])
    base, signed, absolute = band_effects(predict, raw, feats, int(s["repeats"]), int(s["seed"]), int(s["chunk"]))
    patch_mean = raw.reshape(len(raw), raw.shape[1], -1).mean(axis=2)

    folder = Path(out_dir)
    table = pd.DataFrame({"x": xs[keep], "y": ys[keep], "predicted": base})
    for j, band in enumerate(bands):
        table[f"delta_{band}"], table[f"abs_delta_{band}"], table[f"patch_mean_{band}"] = signed[:, j], absolute[:, j], patch_mean[:, j]
    table.to_csv(folder / "band_ablation.csv", index=False)
    imp = importance(absolute, signed, seed=int(s["seed"]))
    imp.insert(0, "band", bands)
    order = np.argsort(-imp["mean_abs_score"].to_numpy(), kind="stable")
    plot_band_importance(imp, bands, order, str(folder / "band_ablation.png"), dpi,
                         heading=f"Band impact — {len(table)} NCR cells, {ts:%Y-%m-%d %H:%M}",
                         xlabel="mean(|change in prediction|) (µg/m³)")
    plot_band_swarm(signed, band_swarm.colour_levels(patch_mean), bands, str(folder / "band_ablation_swarm.png"), dpi,
                    heading=f"Prediction change — {len(table)} NCR cells, {ts:%Y-%m-%d %H:%M}",
                    xlabel="change in the prediction when the band is swapped (µg/m³)", order=order)
    return {"cells_sampled": int(len(flat)), "cells_explained": int(len(table)), "repeats": int(s["repeats"]), "seed": int(s["seed"]),
            "order": [bands[i] for i in order], "bands": imp.to_dict("records"),
            "reading": "change in the predicted PM2.5 (ug/m3) when one band of a cell is replaced by that band of another cell, the rest kept; "
                       "the model's sensitivity to the band, not a real-world effect"}
