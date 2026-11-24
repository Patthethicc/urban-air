"""Geographically weighted conformal prediction (GeoCP) with a spatial split (Validation.txt).

    score_i   = |y_i - yhat_i|                      (ug/m3, calibration rows; the nonconformity score alpha_i)
    w_ij      = K(planar distance(station_i, query cell centre))    (metres, projected CRS)
    q_j       = (1 - alpha) quantile of the scores under the weights NORMALISED TO SUM TO ONE
    interval  = yhat_j +/- q_j, then the lower end is clipped at 0

`finite_sample_correction: true` (geocp.yaml) additionally places the query's own weight (1) at +infinity,
the classical weighted-conformal correction; the default follows the thesis text (plain normalised weights).

Only calibration stations are used here. The test stations are touched once, after the configuration
(kernel, bandwidth) has been picked by leave-one-station-out on the calibration set, and the lower-bound
clipping never changes the coverage check (coverage uses the UNCLIPPED interval).
"""
import numpy as np
import pandas as pd


def kernel_weights(dist: np.ndarray, kernel: str, bandwidth: float) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    if np.isinf(bandwidth):
        return np.ones_like(dist)
    if kernel == "gaussian":
        return np.exp(-0.5 * (dist / bandwidth) ** 2)
    if kernel == "exponential":
        return np.exp(-dist / bandwidth)
    raise ValueError(f"unknown kernel '{kernel}'")


class GeoCP:
    """Fitted on calibration rows: location_id, score. Station coordinates come from `station_xy`."""

    def __init__(self, alpha: float, kernel: str, bandwidth: float, finite_sample_correction: bool = False):
        self.alpha, self.kernel, self.bandwidth = alpha, kernel, bandwidth
        self.correction = bool(finite_sample_correction)

    def fit(self, cal: pd.DataFrame, station_xy: pd.DataFrame) -> "GeoCP":
        """cal: DataFrame(location_id, score). station_xy: DataFrame(location_id, x, y)."""
        order = np.argsort(cal["score"].to_numpy(), kind="stable")
        self.scores = cal["score"].to_numpy(dtype=np.float64)[order]
        loc = cal["location_id"].to_numpy()[order]
        self.locations = np.unique(loc)
        self.loc_idx = np.searchsorted(self.locations, loc)
        xy = station_xy.set_index("location_id").loc[self.locations, ["x", "y"]]
        self.loc_xy = xy.to_numpy(dtype=np.float64)
        return self

    def _threshold(self, total):
        return (1.0 - self.alpha) * (total + (1.0 if self.correction else 0.0))

    def quantile(self, xy, exclude: int | None = None) -> float:
        """Interval half-width at projected point xy; `exclude` drops one calibration station (LOSO)."""
        dist = np.hypot(self.loc_xy[:, 0] - xy[0], self.loc_xy[:, 1] - xy[1])
        w_loc = kernel_weights(dist, self.kernel, self.bandwidth)
        if exclude is not None:
            w_loc = np.where(self.locations == exclude, 0.0, w_loc)
        w = w_loc[self.loc_idx]
        total = w.sum()
        if total <= 0:
            return float("inf")                       # no calibration weight anywhere near the query
        cum = np.cumsum(w)
        hit = np.searchsorted(cum, self._threshold(total) * (1 - 1e-12), side="left")
        return float("inf") if hit >= len(w) else float(self.scores[hit])

    def distance_to_nearest_calibration(self, xy) -> float:
        return float(np.hypot(self.loc_xy[:, 0] - xy[0], self.loc_xy[:, 1] - xy[1]).min())

    def quantiles_grid(self, xy: np.ndarray, n_grid: int = 2000, chunk: int = 20000) -> np.ndarray:
        """Vectorised half-widths for many query points (every cell of a map).

        Exact when there are <= n_grid distinct calibration scores; otherwise the quantile is rounded
        UP to the next of n_grid score quantiles (slightly conservative)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        unique = np.unique(self.scores)
        grid = unique if len(unique) <= n_grid else np.unique(
            np.quantile(self.scores, np.linspace(0, 1, n_grid), method="higher"))
        counts = np.stack([np.searchsorted(np.sort(self.scores[self.loc_idx == l]), grid, side="right")
                           for l in range(len(self.locations))]).astype(np.float64)       # [L, G]
        n_l = counts[:, -1]

        out = np.empty(len(xy))
        for a in range(0, len(xy), chunk):
            part = xy[a:a + chunk]
            dist = np.hypot(part[:, None, 0] - self.loc_xy[None, :, 0], part[:, None, 1] - self.loc_xy[None, :, 1])
            W = kernel_weights(dist, self.kernel, self.bandwidth)                         # [c, L]
            total = W @ n_l
            mass = W @ counts                                                             # [c, G]
            enough = mass >= (self._threshold(total) * (1 - 1e-12))[:, None]
            first = enough.argmax(axis=1)
            out[a:a + chunk] = np.where(enough.any(axis=1) & (total > 0), grid[first], np.inf)
        return out


def scores_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """pred: location_id, y_true, y_pred (ug/m3) -> location_id, score."""
    return pd.DataFrame({"location_id": pred["location_id"].to_numpy(),
                         "score": np.abs(pred["y_true"].to_numpy() - pred["y_pred"].to_numpy())})


def calibration_table(pred: pd.DataFrame, station_xy: pd.DataFrame) -> pd.DataFrame:
    """Georeferenced calibration table (the input GeoCP needs from ANY model, so the same pipeline
    serves XGBoost and CNN-LSTM): location_id, cell_id (raster cell of the patch centre, when known),
    x, y = projected CELL-CENTRE coordinates in metres, timestamp, y_true, y_pred, score = |y - y_hat|."""
    keep = ["location_id", "x", "y"]
    extra = []
    if {"cell_row", "cell_col"} <= set(station_xy.columns):
        extra = ["cell_row", "cell_col"]
    out = pred.merge(station_xy[keep + extra], on="location_id", how="left")
    cols = ["location_id", "x", "y", "timestamp", "y_true", "y_pred"]
    if extra:
        out["cell_id"] = out["cell_row"].astype(int).astype(str) + "_" + out["cell_col"].astype(int).astype(str)
        cols = ["location_id", "cell_id", "x", "y", "timestamp", "y_true", "y_pred"]
    out["score"] = (out["y_true"] - out["y_pred"]).abs()
    return out[cols + ["score"]]


def select_config(cal: pd.DataFrame, station_xy: pd.DataFrame, gcfg: dict):
    """Leave-one-station-out over the calibration stations for every (kernel, bandwidth).

    For each held-out station: interval half-width from the OTHER stations, then coverage of that
    station's own scores. Infinite widths count as covered but are capped at twice the largest score
    when comparing widths. Pick the narrowest configuration whose mean coverage is within
    `coverage_tolerance` of 1 - alpha; if none is, the one with the highest coverage.
    Returns (best: dict, table: DataFrame)."""
    alpha, tol = gcfg["alpha"], gcfg["coverage_tolerance"]
    correction = bool(gcfg.get("finite_sample_correction", False))
    cap = 2.0 * float(cal["score"].max())
    stations = np.unique(cal["location_id"].to_numpy())
    xy = station_xy.set_index("location_id")

    rows = []
    for kernel in gcfg["kernels"]:
        for bw in gcfg["bandwidths_m"]:
            bw = float(bw)
            if np.isinf(bw) and kernel != gcfg["kernels"][0]:
                continue                                  # unweighted baseline only once
            model = GeoCP(alpha, kernel, bw, correction).fit(cal, station_xy)
            covered, widths, n_inf = [], [], 0
            for s in stations:
                q = model.quantile(xy.loc[s, ["x", "y"]].to_numpy(dtype=float), exclude=int(s))
                n_inf += int(np.isinf(q))
                own = cal.loc[cal["location_id"] == s, "score"].to_numpy()
                covered.append(float(np.mean(own <= q)))
                widths.append(2.0 * min(q, cap))
            rows.append({"kernel": "none" if np.isinf(bw) else kernel, "bandwidth_m": bw,
                         "loso_coverage": float(np.mean(covered)), "loso_mean_width": float(np.mean(widths)),
                         "stations_with_inf_interval": n_inf})
    table = pd.DataFrame(rows)

    ok = table[table["loso_coverage"] >= (1.0 - alpha) - tol]
    best_row = (ok.sort_values("loso_mean_width").iloc[0] if len(ok)
                else table.sort_values("loso_coverage", ascending=False).iloc[0])
    kernel = best_row["kernel"] if best_row["kernel"] != "none" else gcfg["kernels"][0]
    best = {"kernel": kernel, "bandwidth_m": float(best_row["bandwidth_m"]), "alpha": alpha,
            "finite_sample_correction": correction,
            "loso_coverage": float(best_row["loso_coverage"]),
            "loso_mean_width": float(best_row["loso_mean_width"]),
            "met_coverage_target": bool(len(ok))}
    return best, table


def apply_intervals(pred: pd.DataFrame, model: GeoCP, station_xy: pd.DataFrame, lower_bound: float = 0.0,
                    zone_threshold_m: float | None = None, ratio_floor: float = 1.0):
    """Attach intervals to a prediction frame (one half-width per location) and summarise (Table 4.2).

    Columns added: q, lo_raw (unclipped), lo (clipped at `lower_bound`), hi, width = hi - lo (published
    interval), covered (checked on the UNCLIPPED interval).

    Zones: stations are split by their distance to the nearest calibration sensor into `urban_core`
    (<= threshold; well sampled) and `peripheral` (> threshold). threshold = zone_threshold_m, or the
    median over the evaluated stations when None."""
    xy = station_xy.set_index("location_id")
    locs = [int(s) for s in pred["location_id"].unique()]
    pos = {s: xy.loc[s, ["x", "y"]].to_numpy(dtype=float) for s in locs}
    q = {s: model.quantile(pos[s]) for s in locs}
    dist = {s: model.distance_to_nearest_calibration(pos[s]) for s in locs}
    threshold = float(zone_threshold_m) if zone_threshold_m is not None else float(np.median(list(dist.values())))
    zone = {s: ("urban_core" if dist[s] <= threshold else "peripheral") for s in locs}

    out = pred.copy()
    out["q"] = out["location_id"].map(q).astype(float)
    out["lo_raw"] = out["y_pred"] - out["q"]
    out["lo"] = np.maximum(out["lo_raw"], lower_bound)
    out["hi"] = out["y_pred"] + out["q"]
    out["width"] = out["hi"] - out["lo"]
    out["covered"] = (out["y_true"] >= out["lo_raw"]) & (out["y_true"] <= out["hi"])
    out["zone"] = out["location_id"].map(zone)

    finite = out[np.isfinite(out["q"])]
    ratio = finite["width"] / np.maximum(finite["y_pred"], ratio_floor)
    per_station = (out.groupby("location_id")
                   .agg(rows=("covered", "size"), coverage=("covered", "mean"), half_width=("q", "first"))
                   .reset_index())
    per_station["distance_to_nearest_calibration_m"] = per_station["location_id"].map(dist)
    per_station["zone"] = per_station["location_id"].map(zone)

    def zone_cov(name):
        sub = out[out["zone"] == name]
        return float(sub["covered"].mean()) if len(sub) else None

    summary = {
        "target_coverage": 1.0 - model.alpha,
        "coverage": float(out["covered"].mean()),
        "mean_width": float(finite["width"].mean()) if len(finite) else float("nan"),
        "median_width": float(finite["width"].median()) if len(finite) else float("nan"),
        "mean_width_to_prediction_ratio": float(ratio.mean()) if len(finite) else float("nan"),
        "coverage_urban_core": zone_cov("urban_core"),
        "coverage_peripheral": zone_cov("peripheral"),
        "zone_threshold_m": threshold,
        "zone_rule": ("urban_core = distance to nearest calibration sensor <= threshold"
                      + ("" if zone_threshold_m is not None else " (threshold = median over the test stations)")),
        "mean_width_unclipped": float((finite["hi"] - finite["lo_raw"]).mean()) if len(finite) else float("nan"),
        "rows_with_inf_interval": int((~np.isfinite(out["q"])).sum()),
        "per_station": per_station.to_dict(orient="records"),
    }
    return out, summary
