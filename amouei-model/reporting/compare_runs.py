"""Compare training runs of different datasets ("arms", each with one or more seeds) on their OUT-OF-FOLD predictions.

  python -m reporting.compare_runs --runs-root runs                      # runs/<arm>/checkpoints/<run>/oof_predictions.csv
  python -m reporting.compare_runs --arm raw=runs/raw/checkpoints --arm from_2023=runs/from_2023/checkpoints

Why a tool: each run's own SUMMARY covers different hours, so two runs cannot be compared from their own numbers. Here every run is
scored on the (station, hour) rows ALL of them have, and on rows where every run agrees on the true value (a reading that one
dataset removed and interpolated, or that cleaning changed, has no common truth). Only out-of-fold predictions of the TRAINING
stations are used, never the test partition, so choosing a dataset does not spend the test set (docs/EXPERIMENT_CHECKLIST.md).

Per arm: RMSE / MAE / R2 / bias, mean and sd over its runs (seeds); the gap to the best arm with a station-cluster bootstrap CI
(whole stations are resampled, because the readings of one station are not independent); `tied` when the CI contains 0 or the gap is
within the seed-to-seed spread. Also reported: whether every run puts every station in the same CV fold (data.split.cv_assignment
by_station makes that true), the weight of the biggest stations, and how many rows each arm lost to the intersection.
"""
import argparse
import json
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from reporting.summary import _table

OOF, MODEL_FILE = "oof_predictions.csv", "best_dual_branch_model.pt"
KEY_SCALE = 10 ** 10                     # key = location_id * KEY_SCALE + seconds since 1970: unique per station-hour, fits in int64


def run_folders(path: Path) -> list[Path]:
    """`path` is one run folder (it holds oof_predictions.csv) or a checkpoints folder of runs."""
    path = Path(path)
    if (path / OOF).is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.is_dir() and (p / OOF).is_file()) if path.is_dir() else []


def discover(runs_root) -> dict:
    """{arm: [run folders]} for runs_root/<arm>/checkpoints/<run>/."""
    root = Path(runs_root)
    arms = {a.name: run_folders(a / "checkpoints") for a in sorted(root.iterdir()) if a.is_dir()}
    return {k: v for k, v in arms.items() if v}


def run_facts(folder: Path) -> dict:
    """Seed, fold assignment and whether the run finished (the OOF file is written before the final fit, so a killed run has one)."""
    path = folder / "run_info.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    status = info.get("status")
    hp = info.get("hyperparameters", {})
    return {"completed": status == "completed" or (status is None and (folder / MODEL_FILE).is_file()), "status": status or "unknown",
            "seed": hp.get("train", {}).get("seed"), "assignment": hp.get("split", {}).get("cv_assignment", "balanced")}


def _read(folder: Path, columns=None) -> pd.DataFrame:
    df = pd.read_csv(folder / OOF, usecols=columns)
    seconds = pd.to_datetime(df["timestamp"]).to_numpy(dtype="datetime64[s]").astype("int64")
    df["key"] = df["location_id"].to_numpy(dtype="int64") * KEY_SCALE + seconds
    if df["key"].duplicated().any():
        raise ValueError(f"{folder / OOF} lists the same station-hour more than once")
    return df.sort_values("key", ignore_index=True)


def _score(sse: np.ndarray, sae: np.ndarray, err: np.ndarray, n: int, sst: float) -> dict:
    return {"rmse": float(np.sqrt(sse.sum() / n)), "mae": float(sae.sum() / n), "r2": float(1 - sse.sum() / sst), "bias": float(err.sum() / n)}


def station_weights(n_stations: int, draws: int, seed: int) -> np.ndarray:
    """[draws, n_stations] bootstrap multiplicities: each draw picks n_stations stations with replacement (whole stations, never rows)."""
    return np.random.default_rng(seed).multinomial(n_stations, np.full(n_stations, 1 / n_stations), size=draws).astype(float)


def compare(arms: dict, include_incomplete: bool = False, bootstrap: int = 2000, seed: int = 42, prefer=None) -> dict:
    runs, skipped = [], []
    for arm, folders in arms.items():
        for folder in folders:
            facts = run_facts(Path(folder))
            if facts["completed"] or include_incomplete:
                runs.append({"arm": arm, "folder": Path(folder), **facts})
            else:
                skipped.append({"arm": arm, "run": Path(folder).name, "status": facts["status"]})
    if not runs:
        raise ValueError("no completed runs to compare")

    # pass 1: the rows every run has (keys only)
    own = [_read(r["folder"], ["location_id", "timestamp"])["key"].to_numpy() for r in runs]
    common = reduce(lambda a, b: np.intersect1d(a, b, assume_unique=True), own)
    if not len(common):
        raise ValueError("the runs share no (station, hour) rows: they cover disjoint periods or stations")

    # pass 2: values at those rows; the true value must be the same in every run
    truth, preds, folds = None, [], []
    alive = np.ones(len(common), dtype=bool)
    for r in runs:
        df = _read(r["folder"])
        at = np.searchsorted(df["key"].to_numpy(), common)
        y = df["y_true"].to_numpy(dtype=float)[at]
        if truth is None:
            truth = y
        alive &= np.isclose(y, truth, rtol=0, atol=1e-6)
        preds.append(df["y_pred"].to_numpy(dtype=float)[at])
        folds.append(df["fold"].to_numpy()[at] if "fold" in df else np.zeros(len(common), dtype=int))
    disagreeing = int((~alive).sum())
    common, truth = common[alive], truth[alive]
    preds, folds = [p[alive] for p in preds], [f[alive] for f in folds]
    if not len(common):
        raise ValueError("the runs share no rows on which they agree about the true value")

    stations, st = np.unique(common // KEY_SCALE, return_inverse=True)
    n_s = np.bincount(st).astype(float)
    n, sst = len(common), float(((truth - truth.mean()) ** 2).sum())
    sse_s, rows = [], []
    for r, p in zip(runs, preds):
        err = p - truth
        sse_s.append(np.bincount(st, err ** 2, minlength=len(stations)))
        rows.append({"arm": r["arm"], "run": r["folder"].name, "seed": r["seed"], "status": r["status"],
                     **_score(sse_s[-1], np.bincount(st, np.abs(err), minlength=len(stations)), err, n, sst)})
    runs_df = pd.DataFrame(rows)

    # arms: mean and sd over seeds, paired station bootstrap against the best arm
    weights = station_weights(len(stations), bootstrap, seed)
    denominator = weights @ n_s
    boot = {a: np.mean([np.sqrt(weights @ sse_s[i] / denominator) for i, r in enumerate(runs) if r["arm"] == a], axis=0) for a in arms
            if any(r["arm"] == a for r in runs)}
    arm_df = runs_df.groupby("arm", sort=False).agg(
        runs=("rmse", "size"), rmse=("rmse", "mean"), rmse_sd=("rmse", lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
        mae=("mae", "mean"), r2=("r2", "mean"), bias=("bias", "mean")).sort_values("rmse").reset_index()
    best = arm_df["arm"].iloc[0]
    sd = dict(zip(arm_df["arm"], arm_df["rmse_sd"]))
    gaps, lows, highs, tied = [], [], [], []
    for _, a in arm_df.iterrows():
        diff = boot[a["arm"]] - boot[best]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        gap = a["rmse"] - arm_df["rmse"].iloc[0]
        spread = np.nanmax([sd[a["arm"]], sd[best]]) if not (np.isnan(sd[a["arm"]]) and np.isnan(sd[best])) else np.nan
        gaps.append(gap), lows.append(lo), highs.append(hi)
        tied.append(bool(a["arm"] != best and (lo <= 0 <= hi or (not np.isnan(spread) and gap <= spread))))
    arm_df = arm_df.assign(gap_to_best=gaps, ci_low=lows, ci_high=highs, tied_with_best=tied)
    arm_df["common_share_of_own_rows"] = [float(np.mean([len(common) / len(o) for o, r in zip(own, runs) if r["arm"] == a])) for a in arm_df["arm"]]

    candidates = [best] + [a for a, t in zip(arm_df["arm"], tied) if t]
    pick = next((a for a in (prefer or []) if a in candidates), best)
    first = np.unique(st, return_index=True)[1]
    labels = np.stack([f[first] for f in folds])
    share = np.sort(n_s)[::-1] / n
    checks = {"runs": len(runs), "arms": int(arm_df["arm"].nunique()), "common_rows": n, "stations": len(stations),
              "rows_dropped_for_truth_disagreement": disagreeing, "stations_in_different_folds_across_runs": int((labels != labels[0]).any(axis=0).sum()),
              "cv_assignment": sorted({r["assignment"] for r in runs}), "largest_station_share": float(share[0]), "top5_station_share": float(share[:5].sum()),
              "smallest_station_rows": int(n_s.min()), "bootstrap": bootstrap, "seed": seed}
    return {"runs": runs_df, "arms": arm_df, "checks": checks, "skipped": skipped, "best": best, "tied": candidates[1:], "recommendation": pick,
            "prefer": list(prefer or [])}


def to_markdown(res: dict) -> str:
    c, arms = res["checks"], res["arms"]
    f = lambda x, d=3: "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
    rows = [[a.arm, a.runs, f(a.rmse), f(a.rmse_sd), f(a.mae), f(a.r2), f(a.bias, 2), f"{a.gap_to_best:+.3f}" if a.arm != res["best"] else "best",
             f"[{a.ci_low:+.3f}, {a.ci_high:+.3f}]" if a.arm != res["best"] else "", "yes" if a.tied_with_best else ("" if a.arm == res["best"] else "no"),
             f"{a.common_share_of_own_rows:.0%}"] for a in arms.itertuples()]
    folds_same = c["stations_in_different_folds_across_runs"] == 0
    lines = [f"# Dataset comparison on out-of-fold predictions ({c['runs']} runs, {c['arms']} arms)", "",
             f"Scored on the {c['common_rows']:,} (station, hour) rows that every run has and agrees on, {c['stations']} training stations. "
             "Out-of-fold only: the test partition is not used.", "",
             _table(["Arm", "Runs", "RMSE", "RMSE sd", "MAE", "R2", "Bias", "Gap to best", "95% CI of gap", "Tied", "Common rows / own"], rows,
                    "lrrrrrrrrrr"), "",
             f"**Best by RMSE: `{res['best']}`.** Tied with it: {', '.join(f'`{a}`' for a in res['tied']) or 'none'}. "
             + (f"With the preference order {res['prefer']}, the choice is **`{res['recommendation']}`**." if res["prefer"] else
                "Apply your tie-break rule among the tied arms."), "",
             "A gap is real only when its CI excludes 0 and it exceeds the seed-to-seed spread (RMSE sd); with one run per arm there is no spread to compare.", "",
             "## Checks", "",
             f"- Same CV folds for every station in every run: **{'yes' if folds_same else 'NO'}**"
             + ("" if folds_same else f" ({c['stations_in_different_folds_across_runs']} of {c['stations']} stations change fold; "
                f"the gaps above also reflect which stations were held out together. Use `data.split.cv_assignment=by_station`.)")
             + f" (assignments used: {', '.join(c['cv_assignment'])})",
             f"- Rows dropped because runs disagree on the true value: {c['rows_dropped_for_truth_disagreement']:,}",
             f"- Largest station is {c['largest_station_share']:.1%} of the common rows, the five largest {c['top5_station_share']:.1%}; smallest has {c['smallest_station_rows']:,} rows",
             f"- Bootstrap: {c['bootstrap']} resamples of whole stations (seed {c['seed']})"]
    if res["skipped"]:
        lines.append("- Skipped (not completed): " + ", ".join(f"{s['arm']}/{s['run']} ({s['status']})" for s in res["skipped"]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", help="folder with <arm>/checkpoints/<run>/ for every arm")
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH", help="one arm: a run folder or a checkpoints folder (repeatable)")
    ap.add_argument("--out-dir", default="comparison")
    ap.add_argument("--prefer", default="", help="comma list, tie-break order among the tied arms (first wins)")
    ap.add_argument("--include-incomplete", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    arms = discover(a.runs_root) if a.runs_root else {}
    for spec in a.arm:
        label, _, path = spec.partition("=")
        found = run_folders(Path(path))
        if not found:
            raise SystemExit(f"--arm {spec}: no {OOF} in that folder or in its subfolders (give a run folder or a checkpoints folder of runs)")
        arms.setdefault(label, []).extend(found)
    if not arms:
        raise SystemExit("give --runs-root or at least one --arm LABEL=PATH that contains oof_predictions.csv")
    res = compare(arms, a.include_incomplete, a.bootstrap, a.seed, [p for p in a.prefer.split(",") if p])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["arms"].to_csv(out / "comparison_arms.csv", index=False)
    res["runs"].to_csv(out / "comparison_runs.csv", index=False)
    text = to_markdown(res)
    (out / "comparison.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
