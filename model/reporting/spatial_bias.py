"""Geostatistical bias detection on out-of-fold residuals (Validation.txt).

    residual = actual - predicted      (so High-High = clusters of systematic UNDER-prediction,
                                        Low-Low = clusters of systematic OVER-prediction)
    1. out-of-fold residuals of the spatial CV on the TRAINING stations only (never calibration or test)
    2. mean residual per geographic grid cell (data/report.yaml bias.cell_size_m; null = raster pixel)
    3. K-nearest-neighbour spatial weights over the cell centres (row-standardised)
    4. Global Moran's I (permutation p-value)
    5. only if the global test is significant (bias.alpha): Anselin's Local Moran's I -> LISA clusters
    6. LISA clusters overlaid on the Master GeoTIFF bands + per-cluster band profile, to see which
       morphology the model struggles with
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reporting.layout import snap_file

QUADRANT = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
CLUSTERS = ["High-High", "Low-Low", "High-Low", "Low-High", "Not significant"]


def cell_geometry(x, y, raster_path: str, cell_size_m: float | None):
    """Cell row/col/centre for projected points, on a grid anchored at the raster's top-left corner."""
    import rasterio

    with rasterio.open(raster_path) as src:
        left, top, res = src.bounds.left, src.bounds.top, float(src.res[0])
    size = float(cell_size_m or res)
    col = np.floor((np.asarray(x) - left) / size).astype(int)
    row = np.floor((top - np.asarray(y)) / size).astype(int)
    return row, col, left + (col + 0.5) * size, top - (row + 0.5) * size, size


def aggregate_residuals_by_cell(oof: pd.DataFrame, stations: pd.DataFrame, raster_path: str,
                                cell_size_m: float | None = None) -> pd.DataFrame:
    """Mean residual per grid cell. oof: location_id, y_true, y_pred; stations: location_id, x, y."""
    df = oof.merge(stations[["location_id", "x", "y"]], on="location_id", how="left")
    df["residual"] = df["y_true"] - df["y_pred"]
    row, col, cx, cy, _ = cell_geometry(df["x"], df["y"], raster_path, cell_size_m)
    df["cell_id"] = [f"{r}_{c}" for r, c in zip(row, col)]
    df["cx"], df["cy"] = cx, cy
    out = (df.groupby("cell_id")
           .agg(mean_residual=("residual", "mean"), n_rows=("residual", "size"),
                n_stations=("location_id", "nunique"),
                stations=("location_id", lambda s: ";".join(str(int(i)) for i in sorted(set(s)))),
                cx=("cx", "first"), cy=("cy", "first"))
           .reset_index())
    return out


def knn_weights(coords: np.ndarray, k: int):
    from libpysal.weights import KNN

    w = KNN.from_array(np.asarray(coords, dtype=float), k=min(int(k), len(coords) - 1))
    w.transform = "r"
    return w


def global_moran(values, w, permutations: int = 999, seed: int = 42, alpha: float = 0.05) -> dict:
    from esda.moran import Moran

    np.random.seed(seed)
    m = Moran(np.asarray(values, dtype=float), w, permutations=permutations)
    p_two = min(1.0, 2.0 * float(m.p_sim))            # esda's p_sim is one-sided; test both tails like the XGBoost side
    return {"I": float(m.I), "expected_I": float(m.EI), "z_sim": float(m.z_sim), "p_sim": p_two,
            "p_sim_one_sided": float(m.p_sim), "permutations": permutations,
            "significant": bool(p_two < alpha), "alpha": alpha}


def local_moran(values, w, permutations: int = 999, seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    """LISA per cell: local I, pseudo p-value, quadrant and cluster label ('Not significant' if p >= alpha)."""
    from esda.moran import Moran_Local

    np.random.seed(seed)
    lm = Moran_Local(np.asarray(values, dtype=float), w, permutations=permutations, alternative="two-sided")
    quadrant = np.array([QUADRANT[int(q)] for q in lm.q])
    return pd.DataFrame({"local_I": lm.Is, "p_sim": lm.p_sim, "quadrant": quadrant,
                         "cluster": np.where(lm.p_sim < alpha, quadrant, "Not significant")})


def band_names_for(raster_path: str, cfg: dict) -> list[str]:
    import rasterio

    with rasterio.open(raster_path) as src:
        if all(src.descriptions):
            return list(src.descriptions)
        order = cfg["data"]["raster"]["band_order"]
        if len(order) == src.count:
            return list(order)
        if src.count < len(order):                  # e.g. the 5-band pilot raster (no sigma_H): leading names
            return list(order[:src.count])
        return [f"band_{i}" for i in range(1, src.count + 1)]


def cell_band_means(raster_path: str, cx, cy, size_m: float) -> np.ndarray:
    """[cells, bands] mean raster value over each cell footprint (outside the raster counts as 0)."""
    import rasterio
    from rasterio.windows import from_bounds

    out = []
    with rasterio.open(raster_path) as src:
        for x, y in zip(cx, cy):
            win = from_bounds(x - size_m / 2, y - size_m / 2, x + size_m / 2, y + size_m / 2, src.transform)
            data = src.read(window=win.round_offsets().round_lengths(), boundless=True, fill_value=0)
            out.append(np.nan_to_num(data.astype(float)).mean(axis=(1, 2)) if data.size else np.zeros(src.count))
    return np.array(out)


def band_profile(cells: pd.DataFrame, means: np.ndarray, names: list[str]) -> list[dict]:
    """Mean of every band per LISA cluster, and how far it sits from the all-cell mean in std units."""
    overall, spread = means.mean(axis=0), means.std(axis=0)
    rows = []
    for cluster in CLUSTERS:
        mask = (cells["cluster"] == cluster).to_numpy()
        if not mask.any():
            continue
        mean = means[mask].mean(axis=0)
        rows.append({"cluster": cluster, "cells": int(mask.sum()),
                     "band_mean": {n: float(v) for n, v in zip(names, mean)},
                     "std_units_from_all_cells": {n: float((m - o) / s) if s > 0 else 0.0
                                                  for n, m, o, s in zip(names, mean, overall, spread)}})
    return rows


def run_spatial_bias(run_id: str, cfg: dict) -> dict:
    """Read the run's out-of-fold predictions, run the analysis, write files into snapshots/<run_id>/ and
    return the summary (also stored in run_info.json)."""
    from reporting import plotters
    from training.run_info import update_run_info, write_json

    paths = cfg["data"]["paths"]
    b = cfg["report"]["bias"]
    run_dir, snap = Path(paths["checkpoints_dir"]) / run_id, Path(paths["snapshots_dir"]) / run_id
    snap.mkdir(parents=True, exist_ok=True)

    oof_path = run_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"{oof_path} missing: train with train.cv.enabled: true")
    oof = pd.read_csv(oof_path)
    partition = json.loads((Path(paths["processed_dir"]) / "partition.json").read_text(encoding="utf-8"))
    foreign = set(oof["location_id"]) - set(partition["train"])
    if foreign:
        raise ValueError(f"out-of-fold residuals contain non-training stations {sorted(foreign)}: the bias "
                         "assessment must not touch calibration or test partitions")
    stations = pd.read_csv(Path(paths["processed_dir"]) / "stations.csv")

    cells = aggregate_residuals_by_cell(oof, stations, paths["master_raster"], b["cell_size_m"])
    summary = {"definition": "residual = actual - predicted; High-High = under-prediction hotspots, "
                             "Low-Low = over-prediction coldspots",
               "source": "out-of-fold predictions of the spatial CV on the training stations",
               "cells": int(len(cells)), "oof_rows": int(len(oof)), "knn_k": int(b["knn_k"])}

    if len(cells) < int(b["min_cells"]) or cells["mean_residual"].nunique() < 2:
        summary["skipped"] = f"fewer than {b['min_cells']} grid cells (or constant residuals); Moran's I not meaningful"
        write_json(snap_file(snap, "spatial_bias.json", write=True), summary)
        update_run_info(run_dir, spatial_bias=summary)
        return summary

    w = knn_weights(cells[["cx", "cy"]].to_numpy(), b["knn_k"])
    glob = global_moran(cells["mean_residual"], w, b["permutations"], b["seed"], b["alpha"])
    summary["global_moran"] = glob

    run_local = glob["significant"] or not b["local_only_if_global_significant"]
    summary["local_moran_run"] = bool(run_local)
    if run_local:
        lisa = local_moran(cells["mean_residual"], w, b["permutations"], b["seed"], b["alpha"])
        cells = pd.concat([cells, lisa], axis=1)
        summary["cluster_counts"] = {c: int((cells["cluster"] == c).sum()) for c in CLUSTERS}
        raster = paths["master_raster"]
        _, _, _, _, size = cell_geometry(cells["cx"], cells["cy"], raster, b["cell_size_m"])
        names = band_names_for(raster, cfg)
        summary["band_profile"] = band_profile(cells, cell_band_means(raster, cells["cx"], cells["cy"], size), names)
        plotters.plot_lisa_on_bands(raster, cells, names, str(snap_file(snap, "lisa_on_bands.png", write=True)))
    else:
        summary["note"] = "global Moran's I not significant: no spatial clustering of residuals, LISA not run"

    cells.to_csv(snap_file(snap, "residuals_by_cell.csv", write=True), index=False)
    plotters.plot_lisa_map(cells, glob, str(snap_file(snap, "lisa_map.png", write=True)))
    write_json(snap_file(snap, "spatial_bias.json", write=True), summary)
    update_run_info(run_dir, spatial_bias={k: v for k, v in summary.items() if k != "band_profile"})
    return summary
