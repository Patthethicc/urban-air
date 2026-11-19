"""The dataset report (--stage dataset): what the model's data is, before any model.

One folder, `dataset_report/` next to the processed data, with DATASET_REPORT.md as the page to read and the tables and figures it uses:

    sensors and coverage      every sensor of the locations table, inside the raster or not, its network (OpenAQ / QC) and partition
    rows                      readings in the file -> usable readings -> hourly hours -> interpolated hours -> hours left missing -> model rows (windows)
    missingness               per sensor: share of its active span with no reading, the longest gap
    temporal availability     per year and network: sensors reporting, station-hours, sensors reporting at the same time; a station x month heatmap
    PM2.5 distribution        percentiles and exceedance shares of the usable hourly readings, per network
    morphology distributions  the five raster bands: over the NCR (10 m pixels and 320 m patch means) and at the sensors' own patches, and how much of the
                              NCR's range the sensors span

The cleaning follows preprocessing/clean_loader.py exactly (same range limits, hourly mean, the same gap cap), so the counts describe the data the model
was built from. Nothing here trains or predicts.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from preprocessing.clean_loader import _fill_short_gaps, find_sensor_file  # noqa: E402

PERCENTILES = (5, 25, 50, 75, 95)
NETWORK_COLOURS = {"QC": "#2B66B1", "OpenAQ": "#D2691E"}
PARTITION_COLOURS = {"train": "#4C9F70", "calibration": "#E9B44C", "test": "#C0392B", "": "#9AA7B0"}


# ------------------------------------------------------------------------------ one sensor
def network_of(path: str | None) -> str:
    """'QC' for a qcdata_<id>_*.csv, 'OpenAQ' for an openaq_<id>_*.csv, '' when the sensor has no file."""
    name = Path(path).name if path else ""
    return "QC" if name.startswith("qcdata") else "OpenAQ" if name.startswith("openaq") else ""


def station_hours(stamps: pd.Series, values: pd.Series, cfg: dict) -> dict:
    """The counts of one sensor from its raw timestamps (naive local time) and values, with the cleaning of clean_loader: values outside
    [clean.min_value, clean.max_value] are dropped, readings are averaged per hour, gaps of at most time.max_gap_hours are interpolated.
    Returns the counts and the hourly series (`observed`: raw hours, `filled`: after interpolation)."""
    d = cfg["data"]
    lo, hi = d["clean"]["min_value"], d["clean"]["max_value"]
    numeric = pd.to_numeric(values, errors="coerce")
    bad = int(numeric.isna().sum())
    out_of_range = int((~numeric.between(lo, hi) & numeric.notna()).sum())
    usable = numeric.where(numeric.between(lo, hi))
    series = pd.Series(usable.to_numpy(), index=pd.DatetimeIndex(stamps)).dropna().sort_index()
    out = {"file_rows": int(len(values)), "bad_rows": bad, "out_of_range_rows": out_of_range, "usable_rows": int(series.notna().sum())}
    if series.empty:
        return {**out, "first_hour": pd.NaT, "last_hour": pd.NaT, "span_hours": 0, "observed_hours": 0, "interpolated_hours": 0,
                "missing_hours": 0, "longest_gap_hours": 0, "observed": series, "filled": series}
    observed = series.resample(d["time"]["freq"]).mean()
    filled = _fill_short_gaps(observed, int(d["time"]["max_gap_hours"]))
    missing = filled.isna()
    run = (missing != missing.shift()).cumsum()
    longest = int(missing.groupby(run).sum().max()) if missing.any() else 0
    return {**out, "first_hour": observed.index[0], "last_hour": observed.index[-1], "span_hours": int(len(observed)),
            "observed_hours": int(observed.notna().sum()), "interpolated_hours": int((filled.notna() & observed.isna()).sum()),
            "missing_hours": int(missing.sum()), "longest_gap_hours": longest, "observed": observed, "filled": filled}


def read_sensor(cfg: dict, row) -> dict:
    """station_hours of one sensor of the locations table ({} counts when it has no CSV)."""
    d = cfg["data"]
    c = d["columns"]
    path = find_sensor_file(d["paths"]["sensor_csv_dir"], int(row[c["group"]]))
    if path is None:
        return {"network": "", "file": "", **station_hours(pd.Series([], dtype="datetime64[ns]"), pd.Series([], dtype=float), cfg)}
    raw = pd.read_csv(path, usecols=[c["timestamp"], c["target"]])
    stamp = pd.to_datetime(raw[c["timestamp"]], utc=True).dt.tz_convert(d["timezone"]).dt.tz_localize(None)
    return {"network": network_of(path), "file": Path(path).name, **station_hours(stamp, raw[c["target"]], cfg)}


# ------------------------------------------------------------------------------ morphology
def block_means(arr: np.ndarray, factor: int) -> np.ndarray:
    """[C, H, W] -> [C, H // factor, W // factor]: the mean of every factor x factor block (the raster cut into patches; leftover edge rows and columns dropped)."""
    c, h, w = arr.shape
    h, w = h // factor * factor, w // factor * factor
    return arr[:, :h, :w].reshape(c, h // factor, factor, w // factor, factor).mean(axis=(2, 4))


def summarize(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    pct = np.percentile(v, PERCENTILES)
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()),
            **{f"p{p}": float(q) for p, q in zip(PERCENTILES, pct)}, "max": float(v.max()), "zero_share": float((v == 0).mean())}


def span_coverage(reference: np.ndarray, sample: np.ndarray) -> dict:
    """How much of the reference distribution the sample spans: the share of the reference inside the sample's min-max and inside its 5th-95th percentile."""
    ref, s = np.asarray(reference, float), np.asarray(sample, float)
    ref, s = ref[np.isfinite(ref)], s[np.isfinite(s)]
    if ref.size == 0 or s.size == 0:
        return {"inside_minmax": np.nan, "inside_p5_p95": np.nan}
    lo, hi = np.percentile(s, [5, 95])
    return {"inside_minmax": float(((ref >= s.min()) & (ref <= s.max())).mean()), "inside_p5_p95": float(((ref >= lo) & (ref <= hi)).mean())}


def morphology(cfg: dict, stations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(summary table, per-station patch means, the arrays for the figure) of the raster bands over the NCR and at the sensors' patches."""
    import rasterio
    from affine import Affine

    from output import cartography
    from preprocessing.spatial_patch_extractor import extract_patches
    from reporting.spatial_bias import band_names_for

    path = cfg["data"]["paths"]["master_raster"]
    names = band_names_for(path, cfg)
    factor = max(1, int(round(cfg["data"]["patch"]["extent_m"] / cfg["derived"]["raster_resolution_m"])))
    boundary = cartography.active_boundary(cfg)
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        if src.nodata is not None:
            data[data == np.float32(src.nodata)] = 0.0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)       # as the patches the model sees (extract_patch)
        transform = src.transform
    blocks = block_means(data, factor)
    block_tr = transform * Affine.scale(factor)
    inside_block = (cartography.inside_mask(block_tr, blocks.shape[1:], boundary) if boundary is not None
                    else np.ones(blocks.shape[1:], bool))
    step = 5                                                                # every 5th pixel (50 m) is plenty for a 10 m distribution
    pixels = data[:, ::step, ::step]
    inside_pixel = (cartography.inside_mask(transform * Affine.scale(step), pixels.shape[1:], boundary) if boundary is not None
                    else np.ones(pixels.shape[1:], bool))

    located = stations[["location_id", "x", "y"]].dropna()
    patches = extract_patches(path, located, cfg)
    station_means = pd.DataFrame({int(i): p.mean(axis=(1, 2)) for i, p in patches.items()}, index=names).T
    station_means.index.name = "location_id"

    rows = []
    for b, name in enumerate(names):
        ncr_blocks, ncr_pixels, at_stations = blocks[b][inside_block], pixels[b][inside_pixel], station_means[name].to_numpy()
        for group, values in ((f"NCR, {cfg['data']['patch']['extent_m']:g} m patch means", ncr_blocks),
                              ("NCR, 10 m pixels (every 5th)", ncr_pixels), ("sensors' own patches (patch mean)", at_stations)):
            rows.append({"band": name, "group": group, **summarize(values)})
        rows.append({"band": name, "group": "NCR range spanned by the sensors", **span_coverage(ncr_blocks, at_stations)})
    arrays = {"names": names, "ncr": {n: blocks[i][inside_block] for i, n in enumerate(names)}, "stations": station_means}
    return pd.DataFrame(rows), station_means.reset_index(), arrays


# ------------------------------------------------------------------------------ figures
def plot_availability(hourly: dict, inventory: pd.DataFrame, path: str, dpi: int = 130) -> str:
    """Station x month heatmap of the share of hours with a usable reading; a bar on the left marks the network."""
    months = pd.period_range(min(s.index.min() for s in hourly.values()), max(s.index.max() for s in hourly.values()), freq="M")
    order = inventory.sort_values(["network", "first_hour"])
    ids = [i for i in order["location_id"] if i in hourly]
    grid = np.full((len(ids), len(months)), np.nan)
    for r, sid in enumerate(ids):
        s = hourly[sid]
        share = s.notna().groupby(s.index.to_period("M")).mean()
        for m, v in share.items():
            grid[r, months.get_loc(m)] = v
    fig, (side, ax) = plt.subplots(1, 2, figsize=(13, 0.13 * len(ids) + 2.2), gridspec_kw={"width_ratios": [0.025, 1], "wspace": 0.01})
    nets = order.set_index("location_id").loc[ids, "network"].tolist()
    side.imshow(np.array([[matplotlib.colors.to_rgb(NETWORK_COLOURS.get(n, "#9AA7B0"))] for n in nets]), aspect="auto")
    side.axis("off")
    image = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ticks = [i for i, m in enumerate(months) if m.month == 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(months[i].year) for i in ticks])
    ax.set_yticks([])
    ax.set_xlabel("month")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01).set_label("share of hours with a reading")
    ax.set_title("Temporal availability per sensor (blue bar: QC network, orange bar: OpenAQ)", fontsize=10, loc="left")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pm25(values_by_network: dict, path: str, dpi: int = 130) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for net, v in values_by_network.items():
        axes[0].hist(np.clip(v, 0, 100), bins=100, density=True, alpha=0.55, label=f"{net} ({len(v):,} h)", color=NETWORK_COLOURS.get(net))
    axes[0].set_xlabel("hourly PM2.5 (µg/m³), clipped at 100")
    axes[0].set_ylabel("density")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    axes[0].set_title("Distribution of usable hourly readings", fontsize=10, loc="left")
    for net, v in values_by_network.items():
        x = np.sort(v)
        axes[1].plot(x, np.arange(1, len(x) + 1) / len(x), label=net, color=NETWORK_COLOURS.get(net))
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("hourly PM2.5 (µg/m³)")
    axes[1].set_ylabel("share of hours at or below")
    axes[1].set_title("Cumulative distribution", fontsize=10, loc="left")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_morphology(arrays: dict, path: str, dpi: int = 130) -> str:
    names = arrays["names"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.6))
    for ax, name in zip(np.atleast_1d(axes), names):
        ncr, st = arrays["ncr"][name], arrays["stations"][name].to_numpy()
        ax.hist(ncr[np.isfinite(ncr)], bins=40, density=True, color="#9AA7B0", alpha=0.85, label="NCR patches")
        ax.hist(st[np.isfinite(st)], bins=20, density=True, color="#2B66B1", alpha=0.55, label="sensors")
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("The five bands: 320 m patch means over the NCR against the sensors' patches", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------ the report
def _table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("" if pd.isna(v) else floatfmt.format(v) if isinstance(v, (float, np.floating)) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def run(cfg: dict, out_dir: str | Path) -> dict:
    """Write the report, tables and figures into `out_dir`; returns the headline numbers."""
    import json

    from preprocessing.coverage import all_sensor_locations, coverage_report

    d = cfg["data"]
    c = d["columns"]
    processed = Path(d["paths"]["processed_dir"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sensors = all_sensor_locations(cfg).rename(columns={c["group"]: "location_id", c["x"]: "x", c["y"]: "y"})
    cov = coverage_report(cfg)
    import rasterio
    with rasterio.open(d["paths"]["master_raster"]) as src:
        b = src.bounds
    sensors["inside_raster"] = sensors["x"].between(b.left, b.right) & sensors["y"].between(b.bottom, b.top)

    partition = {}
    if (processed / "partition.json").is_file():
        for name, ids in json.loads((processed / "partition.json").read_text(encoding="utf-8")).items():
            partition.update({int(i): name for i in ids})
    model_rows = {}
    for name in ("train", "calibration", "test"):
        meta = processed / f"meta_{name}.csv"
        if meta.is_file():
            for sid, n in pd.read_csv(meta, usecols=["location_id"])["location_id"].value_counts().items():
                model_rows[int(sid)] = model_rows.get(int(sid), 0) + int(n)

    rows, hourly, observed_values = [], {}, {}
    for r in sensors.to_dict("records"):
        info = read_sensor(cfg, r) if r["inside_raster"] else {"network": network_of(find_sensor_file(d["paths"]["sensor_csv_dir"], int(r["location_id"]))),
                                                              "file": "", **station_hours(pd.Series([], dtype="datetime64[ns]"), pd.Series([], dtype=float), cfg)}
        if r["inside_raster"] and info["observed_hours"]:
            hourly[int(r["location_id"])] = info["observed"]
            observed_values.setdefault(info["network"], []).append(info["observed"].dropna().to_numpy())
        rows.append({"location_id": int(r["location_id"]), "name": r["name"], "network": info["network"], "inside_raster": bool(r["inside_raster"]),
                     "partition": partition.get(int(r["location_id"]), ""),
                     **{k: info[k] for k in ("file_rows", "bad_rows", "out_of_range_rows", "usable_rows", "first_hour", "last_hour", "span_hours",
                                              "observed_hours", "interpolated_hours", "missing_hours", "longest_gap_hours")},
                     "model_rows": model_rows.get(int(r["location_id"]), 0)})
    inv = pd.DataFrame(rows)
    inv["missing_pct"] = np.where(inv["span_hours"] > 0, 100 * (inv["span_hours"] - inv["observed_hours"]) / inv["span_hours"].clip(lower=1), np.nan)
    inv.to_csv(out / "dataset_sensors.csv", index=False)
    used = inv[inv["inside_raster"] & (inv["observed_hours"] > 0)]

    # rows: file -> usable -> hours -> model rows
    totals = {"sensors in the locations table": len(inv), "sensors inside the raster": int(inv["inside_raster"].sum()),
              "sensors with readings (inside the raster)": len(used), "readings in the files": int(used["file_rows"].sum()),
              "not numeric": int(used["bad_rows"].sum()), "outside the allowed range": int(used["out_of_range_rows"].sum()),
              "usable readings": int(used["usable_rows"].sum()), "hours with a reading (hourly mean)": int(used["observed_hours"].sum()),
              "hours filled by interpolation (gaps up to the cap)": int(used["interpolated_hours"].sum()),
              "hours with a value after interpolation": int(used["observed_hours"].sum() + used["interpolated_hours"].sum()),
              "hours left missing in the active spans": int(used["missing_hours"].sum()),
              "model rows (windows with every lag, the target and the spatial lag)": int(used["model_rows"].sum())}
    totals["hours without a model row (next to an outage, or no spatial lag)"] = (totals["hours with a value after interpolation"]
                                                                                  - totals["model rows (windows with every lag, the target and the spatial lag)"])
    rows_table = pd.DataFrame({"step": list(totals), "count": list(totals.values())})
    rows_table.to_csv(out / "dataset_rows.csv", index=False)

    # temporal availability by year: sensors reporting per network, station-hours, sensors reporting at the same hour
    year_rows = []
    for sid, s_ in hourly.items():
        net = inv.loc[inv["location_id"] == sid, "network"].iloc[0]
        have = s_.dropna()
        for year, grp in have.groupby(have.index.year):
            year_rows.append({"year": str(int(year)), "network": net, "location_id": sid, "hours": int(len(grp))})
    yearly = pd.DataFrame(year_rows)
    by_year = pd.DataFrame(columns=["year", "sensors", "QC sensors", "OpenAQ sensors", "station-hours", "mean sensors per hour", "max sensors per hour"])
    if len(yearly):
        per = yearly.groupby("year").agg(sensors=("location_id", "nunique"), hours=("hours", "sum"))
        nets = yearly.groupby(["year", "network"])["location_id"].nunique().unstack(fill_value=0)
        concurrent = pd.concat([s_.notna() for s_ in hourly.values()], axis=1).sum(axis=1)
        concurrent = concurrent[concurrent > 0]
        conc = concurrent.groupby(concurrent.index.year.astype(str)).agg(["mean", "max"])
        by_year = pd.DataFrame({"year": per.index, "sensors": per["sensors"].to_numpy(),
                                "QC sensors": [int(nets.loc[y, "QC"]) if "QC" in nets else 0 for y in per.index],
                                "OpenAQ sensors": [int(nets.loc[y, "OpenAQ"]) if "OpenAQ" in nets else 0 for y in per.index],
                                "station-hours": per["hours"].to_numpy(), "mean sensors per hour": [round(float(conc.loc[y, "mean"]), 1) for y in per.index],
                                "max sensors per hour": [int(conc.loc[y, "max"]) for y in per.index]})
    by_year.to_csv(out / "dataset_yearly.csv", index=False)

    # PM2.5 distribution
    dist_rows = []
    for net, parts in observed_values.items():
        v = np.concatenate(parts)
        dist_rows.append({"network": net, "hours": len(v), "mean": v.mean(), "sd": v.std(), **{f"p{p}": np.percentile(v, p) for p in (5, 25, 50, 75, 95, 99)},
                          "max": v.max(), "share_above_35": (v > 35).mean() * 100, "share_above_50": (v > 50).mean() * 100})
    values_all = np.concatenate([np.concatenate(p) for p in observed_values.values()]) if observed_values else np.array([])
    if len(values_all):
        dist_rows.append({"network": "all", "hours": len(values_all), "mean": values_all.mean(), "sd": values_all.std(),
                          **{f"p{p}": np.percentile(values_all, p) for p in (5, 25, 50, 75, 95, 99)}, "max": values_all.max(),
                          "share_above_35": (values_all > 35).mean() * 100, "share_above_50": (values_all > 50).mean() * 100})
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out / "dataset_pm25_distribution.csv", index=False)

    # morphology
    msummary, mstations, arrays = morphology(cfg, sensors[sensors["inside_raster"]])
    msummary.to_csv(out / "dataset_morphology_summary.csv", index=False)
    mstations = mstations.merge(inv[["location_id", "name", "network", "partition"]], on="location_id", how="left")
    mstations.to_csv(out / "dataset_morphology_stations.csv", index=False)

    figs = {}
    if hourly:
        figs["availability"] = plot_availability(hourly, inv[inv["inside_raster"]], str(out / "fig_availability.png"))
        figs["pm25"] = plot_pm25({k: np.concatenate(v) for k, v in observed_values.items()}, str(out / "fig_pm25_distribution.png"))
    figs["morphology"] = plot_morphology(arrays, str(out / "fig_morphology.png"))
    from explain.morphology_sheet import plot_morphology_sheet
    figs["morphology_maps"] = plot_morphology_sheet(d["paths"]["master_raster"], cfg, [], str(out / "fig_morphology_maps.png"), arrays["names"],
                                                    stations=sensors[sensors["inside_raster"]], caption=f"{Path(d['paths']['master_raster']).name}; city boundaries: GADM 4.1.", dpi=130)

    span = f"{used['first_hour'].min():%Y-%m-%d} to {used['last_hour'].max():%Y-%m-%d}" if len(used) else "no data"
    outside_names = inv[~inv["inside_raster"]][["location_id", "name"]]
    spanned = msummary[msummary["group"] == "NCR range spanned by the sensors"][["band", "inside_minmax", "inside_p5_p95"]].copy()
    spanned[["inside_minmax", "inside_p5_p95"]] = spanned[["inside_minmax", "inside_p5_p95"]] * 100
    spanned.columns = ["band", "% of NCR patches inside the sensors' min-max", "% inside the sensors' 5th-95th percentile"]
    stats = msummary[msummary["group"] != "NCR range spanned by the sensors"].drop(columns=["zero_share"]).copy()
    stats["zero %"] = msummary.loc[stats.index, "zero_share"] * 100
    stats["n"] = stats["n"].astype(int)
    cols = ["band", "group", "n", "mean", "sd", "min", "p5", "p25", "p50", "p75", "p95", "max", "zero %"]
    ncr_line = ""
    if cov.get("boundary"):
        gaps = ", ".join(f"{km:.1f} km beyond the {side} edge" for side, km in cov["boundary"]["missing_km"].items())
        ncr_line = f" The raster covers {cov['boundary']['covered_fraction']:.1%} of the NCR boundary" + (f" (the boundary extends {gaps})." if gaps else ".")

    by_net = used.groupby("network").agg(sensors=("location_id", "count"), usable_readings=("usable_rows", "sum"), observed_hours=("observed_hours", "sum"),
                                         median_missing_pct=("missing_pct", "median"), first=("first_hour", "min"), last=("last_hour", "max")).reset_index()
    by_net["first"], by_net["last"] = by_net["first"].dt.strftime("%Y-%m-%d"), by_net["last"].dt.strftime("%Y-%m-%d")
    by_part = used[used["partition"] != ""].groupby("partition").agg(sensors=("location_id", "count"), model_rows=("model_rows", "sum")).reset_index()

    no_rows = used[(used["model_rows"] == 0) & (used["partition"] != "")]
    no_rows_line = ("Sensors with readings but no model rows: " + "; ".join(f"{r.name} ({r.location_id}, {r.usable_rows:,} readings)" for r in no_rows.itertuples())
                    + ". Their cell has no spatial lag for the hours they report, so they cannot be scored.") if len(no_rows) else ""
    md = [f"# Dataset report", f"*{Path(d['paths']['master_raster']).name} · readings {span} · generated by `--stage dataset`*", "",
          "## 1. In one paragraph",
          f"{len(used)} sensors inside the raster have readings ({', '.join(f'{n} {k}' for k, n in used['network'].value_counts().items())}), covering {span}. "
          f"They give {totals['usable readings']:,} usable readings, {totals['hours with a reading (hourly mean)']:,} hours with a reading, "
          f"{totals['hours with a value after interpolation']:,} hours with a value once short gaps are filled, and "
          f"{totals['model rows (windows with every lag, the target and the spatial lag)']:,} model rows. "
          f"{int((~inv['inside_raster']).sum())} sensor(s) of the locations table lie outside the raster and are skipped.{ncr_line}", "",
          "## 2. Sensors and coverage", _table(by_net),
          ("Outside the raster (skipped): " + "; ".join(f"{r.name} ({r.location_id})" for r in outside_names.itertuples()) + ".") if len(outside_names) else "Every sensor of the locations table is inside the raster.", "",
          "Sensors and model rows per partition (the processed folder this report read):" if len(by_part) else "No partition file was found, so the partition is not reported.",
          _table(by_part) if len(by_part) else "", no_rows_line, "", "Every sensor: `dataset_sensors.csv`.", "",
          "## 3. From the files to the model rows", _table(rows_table),
          "Readings are cleaned as in `preprocessing/clean_loader.py`: values outside the allowed range are dropped, readings are averaged per hour, and gaps of at most "
          f"{d['time']['max_gap_hours']} hours are filled by linear interpolation; longer gaps stay missing. A model row needs every lag, the target and the spatial lag, "
          "so rows are lost next to every outage.", "",
          "## 4. Missingness", f"Share of a sensor's active span (first to last reading) with no reading: median {used['missing_pct'].median():.1f} %, "
          f"worst {used['missing_pct'].max():.1f} %; longest single gap {int(used['longest_gap_hours'].max()):,} hours. Per sensor: `dataset_sensors.csv` (`missing_pct`, `longest_gap_hours`).", "",
          "## 5. Temporal availability", _table(by_year), "![availability](fig_availability.png)", "",
          "## 6. PM2.5 distribution (usable hourly readings)", _table(dist), "![pm25](fig_pm25_distribution.png)", "",
          "## 7. The morphology bands",
          "Distribution of each band over the NCR (10 m pixels, and 320 m patch means, the scale the CNN sees) and at the sensors' own patches. NoData counts as 0, as in the model's patches; a 0 is a pixel with no building (or no data), which is common at 10 m, so `zero %` is part of the picture.",
          _table(stats[cols]), "How much of the NCR's range do the sensors span?", _table(spanned),
          "A low share means the sensors sit in a narrow part of the morphology the map is asked to cover: the model is then extrapolating there.", "![morphology](fig_morphology.png)", "",
          "The five bands as maps over the NCR, in quantile classes (the same sheet the Grad-CAM explanation uses, here without hot and cold cells):", "![the five bands](fig_morphology_maps.png)", "",
          "## 8. Files", "`dataset_sensors.csv`, `dataset_rows.csv`, `dataset_yearly.csv`, `dataset_pm25_distribution.csv`, `dataset_morphology_summary.csv`, "
          "`dataset_morphology_stations.csv`, `fig_availability.png`, `fig_pm25_distribution.png`, `fig_morphology.png`, `fig_morphology_maps.png`."]
    (out / "DATASET_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"sensors": int(len(used)), "span": span, "totals": totals, "files": figs}
