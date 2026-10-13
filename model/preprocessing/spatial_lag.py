"""KNN spatial lag: what the k nearest stations read at the same hour (Preprocessing.txt, "Spatial Lag").

Leakage rules (the part that is easy to get wrong):
  * the reference set is the PROPER-TRAINING stations only, so a calibration or test station never
    contributes its own readings to anyone's feature;
  * a station is never its own neighbour, including training stations;
  * neighbours are looked up at the same hour as the lag step they enrich (t-3, t-2, t-1);
  * inside cross-validation the reference set shrinks to that fold's FIT stations
    (training/fold_features.py recomputes the lag with `lag_matrix` below).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def station_coordinates(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per location with projected x, y (EPSG:32651). Columns: location_id, x, y."""
    c = cfg["data"]["columns"]
    out = df.groupby(c["group"], sort=True)[[c["x"], c["y"]]].first().reset_index()
    out.columns = ["location_id", "x", "y"]
    out["location_id"] = out["location_id"].astype("int64")
    return out


def neighbour_table(coords: pd.DataFrame, reference_ids, cfg: dict) -> dict:
    """{location_id: (neighbour_ids[<=k], weights[<=k])} for every station in `coords`: the nearest
    reference stations excluding itself. Weights are inverse-distance (or uniform) and sum to 1."""
    sl = cfg["data"]["spatial_lag"]
    ref = coords[coords["location_id"].isin(list(reference_ids))].reset_index(drop=True)
    tree = cKDTree(ref[["x", "y"]].to_numpy())
    ref_ids = ref["location_id"].to_numpy()
    k = int(sl["k"])
    kq = min(k + 1, len(ref))

    table = {}
    for row in coords.itertuples(index=False):
        dist, idx = tree.query([row.x, row.y], k=kq)
        dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
        ids = ref_ids[idx]
        keep = ids != row.location_id
        ids, dist = ids[keep][:k], dist[keep][:k]
        if sl["weighting"] == "inverse_distance":
            w = 1.0 / np.maximum(dist, 1.0)
        else:
            w = np.ones_like(dist)
        table[int(row.location_id)] = (ids.astype("int64"), w / w.sum() if len(w) else w)
    return table


def nearest_station_weights(points_xy: np.ndarray, ref_xy: np.ndarray, cfg: dict):
    """For arbitrary query points (e.g. every raster cell): indices into `ref_xy` of the k nearest
    stations and their weights (inverse distance with a 1 m floor, or uniform), rows sum to 1.
    Unlike neighbour_table nothing is excluded: a cell that contains a sensor may use it."""
    sl = cfg["data"]["spatial_lag"]
    k = min(int(sl["k"]), len(ref_xy))
    dist, idx = cKDTree(np.asarray(ref_xy, dtype=float)).query(np.asarray(points_xy, dtype=float), k=k)
    dist, idx = dist.reshape(len(points_xy), k), idx.reshape(len(points_xy), k)
    w = 1.0 / np.maximum(dist, 1.0) if sl["weighting"] == "inverse_distance" else np.ones_like(dist)
    return idx, w / w.sum(axis=1, keepdims=True)


def lag_matrix(values: np.ndarray, column_ids, table: dict, target_ids) -> np.ndarray:
    """values [H, S] = PM2.5 per hour (rows) and station (columns, ids in `column_ids`).
    Returns [H, len(target_ids)]: weighted mean of each target's neighbours at every hour.
    Neighbours without a reading that hour are dropped and the weights renormalised; NaN if none."""
    col = {int(c): i for i, c in enumerate(column_ids)}
    out = np.full((values.shape[0], len(target_ids)), np.nan)
    for j, sid in enumerate(target_ids):
        nb, w = table[int(sid)]
        if len(nb) == 0:
            continue
        vals = values[:, [col[int(n)] for n in nb]].astype(float)
        seen = ~np.isnan(vals)
        num = np.where(seen, vals, 0.0) @ w
        den = seen @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = np.where(den > 0, num / den, np.nan)
    return out


def add_spatial_lag(df: pd.DataFrame, partition: dict, cfg: dict) -> pd.DataFrame:
    """Add column `spatial_lag` = weighted mean of the neighbours' PM2.5 at the same timestamp, with
    the training partition as reference set. NaN when no neighbour reported (the window builder then
    drops the sample). `partition` comes from spatial_partitioner."""
    c = cfg["data"]["columns"]
    g, ts, tg = c["group"], c["timestamp"], c["target"]

    table = neighbour_table(station_coordinates(df, cfg), partition["train"], cfg)
    wide = df.pivot(index=ts, columns=g, values=tg)
    targets = list(table)
    lag_wide = pd.DataFrame(lag_matrix(wide.to_numpy(), wide.columns, table, targets),
                            index=wide.index, columns=targets)
    lag_wide.columns.name = g
    long = lag_wide.reset_index().melt(id_vars=ts, var_name=g, value_name="spatial_lag")
    long[g] = long[g].astype(df[g].dtype)
    return df.merge(long, on=[g, ts], how="left")


def gather_row_lags(lag_mat: np.ndarray, col_pos: np.ndarray, t_idx: np.ndarray, lags) -> np.ndarray:
    """[N, T]: for sample i the lag value of its own station (column col_pos[i]) at hours t_idx[i] - L
    for each lag L (oldest first), i.e. the same values `add_lags` would have shifted in."""
    return np.stack([lag_mat[t_idx - int(L), col_pos] for L in lags], axis=1)
