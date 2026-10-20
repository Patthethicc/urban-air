"""Human-readable results: SUMMARY.md for a snapshot, the block printed when training ends, and the headline printed
after `report`. All of them are built from the numbers in metrics_summary.json / run_info.json, nothing is recomputed.
"""
from datetime import datetime

import numpy as np

from reporting.layout import describe

UG = "µg/m³"


def _pct(x, digits=1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f} %"


def _num(x, digits=2, sign=False) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def _table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    rule = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _plain_table(headers: list[str], rows: list[list]) -> str:
    """Aligned columns for the terminal (the Markdown tables are for SUMMARY.md)."""
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    line = lambda r: "  " + "   ".join(c.rjust(w) for c, w in zip(r, widths))
    return "\n".join([line(headers), "  " + "   ".join("-" * w for w in widths)] + [line(r) for r in cells])


def _tolerances(measure: dict) -> list[tuple[str, float]]:
    out = []
    for k, v in measure.items():
        if k.startswith("within_") and k != "within_one":
            out.append((k[len("within_"):], v))
    return out


def _model_line(info: dict) -> str:
    h, p = info["hyperparameters"], info.get("parameters", {})
    spatial = h["model"]["spatial_cnn"].get("enabled", True)
    tuned = "yes" if info.get("tuned_from") else "no"
    return (f"dual-branch CNN-LSTM, {p.get('total', 0):,} parameters, spatial (CNN) branch "
            f"{'ON' if spatial else 'OFF (temporal-only baseline)'}, batch size {h['train']['batch_size']}, tuned: {tuned}")


# ------------------------------------------------------------------------------ printed when training ends
def training_summary_text(info: dict) -> str:
    cv = info.get("cv", [])
    d = info.get("data", {})
    rows, stations = d.get("rows_per_partition", {}), d.get("stations_per_partition", {})
    bar = "=" * 74
    out = [bar, f"TRAINING SUMMARY   run {info['run_id']}", bar]
    if rows:
        out.append("Data    " + ", ".join(f"{rows[k]:,} {k} rows ({stations.get(k, '?')} stations)"
                                         for k in ("train", "calibration", "test") if k in rows))
    out.append("Model   " + _model_line(info))
    if cv:
        out += ["", "Cross-validation: every fold's stations were never seen while that fold was trained",
                _plain_table(["fold", "held-out stations", f"RMSE ({UG})", f"MAE ({UG})", "R²", "best epoch"],
                             [[c["fold"], len(c.get("val_stations") or []), _num(c["metrics"]["rmse"]), _num(c["metrics"]["mae"]),
                               _num(c["metrics"]["r2"], 3), f"{c['best_epoch']} of {c['epochs_run']}"] for c in cv])]
        for key, label in (("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²")):
            v = [c["metrics"][key] for c in cv]
            out.append(f"  mean ± sd   {label:<5}{np.mean(v):.3f} ± {np.std(v):.3f}")
    ff = info.get("final_fit")
    if ff:
        out += ["", f"Final model: fitted on all training stations for {ff['epochs']} epoch(s) "
                    f"({ff['epochs_source']}), last training loss {ff['last_train_loss']:.4f}."]
    out += ["", "These are validation figures. The accuracy on the TEST stations (never seen, not used to pick anything)",
            "and the uncertainty intervals come next:",
            "    python run_pipeline.py --stage report        (writes snapshots/<run>/SUMMARY.md)", bar]
    return "\n".join(out)


# ------------------------------------------------------------------------------ printed after `report`
def headline_text(summary: dict, snapshot_dir) -> str:
    acc = summary["accuracy"]
    t = acc["measures"]["test"]
    lines = [f"ACCURACY on {t['stations']} test stations never seen in training ({t['rows']:,} station-hours)",
             f"  R² {_num(t['r2'], 3)}    RMSE {_num(t['rmse'])} {UG}    MAE {_num(t['mae'])} {UG}    bias {_num(t['bias'], 2, True)} {UG}"]
    tol = "   ".join(f"within ±{k} {UG}: {_pct(v)}" for k, v in _tolerances(t))
    if "aqi_exact" in t:
        tol += (f"   same AQI band: {_pct(t['aqi_exact'])} (always guessing '{t['aqi_majority_band']}' "
                f"would give {_pct(t['aqi_majority_baseline'])})")
    lines.append("  " + tol.strip())
    y = acc.get("yardsticks")
    if y:
        lines.append(f"  vs repeating the previous hour: RMSE {_num(y['model']['rmse'])} against {_num(y['persistence']['rmse'])}"
                     f"  (skill {y['skill_vs_persistence']:+.0%})")
    g = summary.get("geocp", {}).get("test")
    if g:
        lines.append(f"UNCERTAINTY  the {g['target_coverage']:.0%} intervals contain the observation {_pct(g['coverage'])} of the time, "
                     f"mean width {_num(g['mean_width'])} {UG}")
    lines.append(f"Full report: {snapshot_dir}/SUMMARY.md")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ SUMMARY.md
def _cautions(summary: dict, info: dict) -> list[str]:
    notes = []
    acc = summary["accuracy"]
    t = acc["measures"]["test"]
    ov = [o for o in info.get("overrides", []) if o.startswith(("train.epochs", "train.final_epochs"))]
    if ov:
        notes.append(f"This run used {', '.join(ov)}: a short run, not a model trained to convergence.")
    if t["stations"] < 20:
        notes.append(f"Only {t['stations']} test stations: the headline numbers depend on which stations were drawn, "
                     "and the per-station figures are noisy. Read differences of a fraction of a µg/m³ as noise.")
    y = acc.get("yardsticks")
    if y and y["skill_vs_persistence"] < 0:
        notes.append("The model is NOT more accurate than repeating the previous hour at stations that have a history. "
                     "It can still be useful where there is no history, which persistence cannot do.")
    g = summary.get("geocp", {}).get("test")
    if g and g["coverage"] < g["target_coverage"] - 0.02:
        notes.append(f"The intervals under-cover on the test stations ({_pct(g['coverage'])} against a "
                     f"{g['target_coverage']:.0%} target). They are narrower than the model's real errors warrant.")
    if "aqi_exact" in acc["measures"]["test"]:
        notes.append("AQI bands use the breakpoints in configs/output.yaml; only the Good / Fair / Unhealthy-for-sensitive-groups "
                     "limits are confirmed there (see the note in that file).")
    notes.append("R² and RMSE pool all station-hours of all stations. Error is not equal across hours: PM2.5 spikes are "
                 "rare and dominate RMSE.")
    return notes


def build_markdown(summary: dict, info: dict, files: list[str]) -> str:
    acc = summary["accuracy"]
    m = acc["measures"]
    t = m["test"]
    geo = summary.get("geocp", {})
    created = info.get("created", "")[:19].replace("T", " ")
    L = [f"# Results of run {summary['run_id']}",
         f"*Trained {created} · {_model_line(info)} · generated {datetime.now():%Y-%m-%d %H:%M}*", ""]

    # ---- 1. headline
    L += ["## 1. Headline: accuracy on the test stations",
          f"{t['stations']} stations and {t['rows']:,} station-hours that were **never seen in training** and were not used to "
          "choose any setting. This is the number to quote.", ""]
    rows = [["R²", _num(t["r2"], 3), "share of the variation in PM2.5 the model explains (1 = perfect, 0 = no better than the average level)"],
            ["RMSE", f"{_num(t['rmse'])} {UG}", "typical error; large misses count more"],
            ["MAE", f"{_num(t['mae'])} {UG}", "average size of an error"],
            ["Bias", f"{_num(t['bias'], 2, True)} {UG}", "average predicted − observed: below 0 = under-predicts, above 0 = over-predicts"]]
    for k, v in _tolerances(t):
        rows.append([f"Within ±{k} {UG}", _pct(v), f"share of predictions whose error is at most {k} {UG}"])
    if "aqi_exact" in t:
        rows.append(["Same AQI band", _pct(t["aqi_exact"]),
                     f"predicted category equals the observed one. Guessing '{t['aqi_majority_band']}' every time scores "
                     f"{_pct(t['aqi_majority_baseline'])}, so judge it against that"])
        rows.append(["Within one AQI band", _pct(t["aqi_within_one"]), "off by at most one category"])
    g = geo.get("test")
    if g:
        rows.append([f"{g['target_coverage']:.0%} interval coverage", _pct(g["coverage"]),
                     f"share of observations inside the GeoCP interval (target {g['target_coverage']:.0%})"])
        rows.append(["Mean interval width", f"{_num(g['mean_width'])} {UG}", "upper − lower; narrower is more useful if coverage holds"])
    L += [_table(["Measure", "Value", "Meaning"], rows, "lll"), ""]
    sup = [f"{k.replace('_', ' ')} {_num(v, 3)}" for k, v in t.items() if k in ("ia", "pearson_r")]
    if sup:
        L += [f"Also reported (thesis supplementary metrics): {', '.join(sup)}.", ""]

    # ---- 2. yardsticks
    y = acc.get("yardsticks")
    L += ["## 2. Is that good? Compared with simple rules"]
    if y:
        L += [f"On the {y['rows']:,} test rows whose previous hour is also in the data ({_pct(y['share_of_test_rows'], 0)}). "
              "Persistence needs the station's own recent history, which a place without a sensor does not have; the model is built for those places.", "",
              _table(["Predictor", f"RMSE ({UG})", f"MAE ({UG})", "R²"],
                     [["**The model**", _num(y["model"]["rmse"]), _num(y["model"]["mae"]), _num(y["model"]["r2"], 3)],
                      ["Repeat the previous hour (persistence)", _num(y["persistence"]["rmse"]), _num(y["persistence"]["mae"]),
                       _num(y["persistence"]["r2"], 3)],
                      ["Always predict the average level", _num(y["mean_predictor"]["rmse"]), _num(y["mean_predictor"]["mae"]), "0.000"]]),
              "", f"Skill against persistence: **{y['skill_vs_persistence']:+.0%}** (1 − MSE of the model ÷ MSE of persistence; "
                  "above 0 the model is more accurate, 0 is a tie).", ""]
    else:
        L += ["Not computed (switched off, or the predictions have no timestamps).", ""]

    # ---- 3. partitions
    L += ["## 3. Does the accuracy hold across the data splits?",
          "Stations are split, not hours, so every row below is about places the model had not seen when it made the prediction.", ""]
    names = {"cross_validation": ("Cross-validation", "training stations, each held out in turn (out-of-fold)"),
             "calibration": ("Calibration", "stations used only to build the intervals"),
             "test": ("**Test**", "final, untouched")}
    prow = []
    for key, (label, what) in names.items():
        if key in m:
            p = m[key]
            prow.append([label, what, p["stations"], f"{p['rows']:,}", _num(p["rmse"]), _num(p["mae"]), _num(p["r2"], 3),
                         _num(p["bias"], 2, True)])
    L += [_table(["Split", "What it is", "Stations", "Rows", f"RMSE ({UG})", f"MAE ({UG})", "R²", "Bias"], prow, "llrrrrrr"), ""]
    if "cross_validation" in m and m["cross_validation"]["rmse"] > 0:
        d = t["rmse"] / m["cross_validation"]["rmse"] - 1
        folds = [c["metrics"]["rmse"] for c in info.get("cv", [])]
        spread = (f" The cross-validation folds themselves differ (RMSE {min(folds):.2f} to {max(folds):.2f}), so that "
                  "reference figure is uncertain too." if len(folds) > 1 and max(folds) > 1.25 * min(folds) else "")
        if abs(d) <= 0.15:
            L += [f"Test RMSE is {abs(d):.0%} {'above' if d > 0 else 'below'} the cross-validation RMSE: the accuracy found "
                  "during validation carries over to unseen stations." + spread, ""]
        elif d > 0:
            L += [f"Test RMSE is {d:.0%} ABOVE the cross-validation RMSE: validation was optimistic, which is worth "
                  "investigating (too few stations, or test stations unlike the training ones)." + spread, ""]
        else:
            L += [f"Test RMSE is {-d:.0%} below the cross-validation RMSE: the test stations were easier than the held-out "
                  "training stations, so there is no sign of over-fitting to the validation process." + spread, ""]

    # ---- 4. stations
    bs = acc.get("by_station", [])
    if bs:
        L += ["## 4. Accuracy by test station (largest error first)", ""]
        srows = []
        for r in bs:
            srows.append([r.get("name") or r["location_id"], r.get("network", ""), f"{r['rows']:,}", _num(r["rmse"]), _num(r["mae"]),
                          _num(r["r2"], 2), _num(r["bias"], 2, True), _pct(r.get("geocp_coverage"), 0) if "geocp_coverage" in r else ""])
        L += [_table(["Station", "Network", "Rows", f"RMSE ({UG})", f"MAE ({UG})", "R²", "Bias", "Interval coverage"], srows, "llrrrrrr"), "",
              "![accuracy by station](plots/accuracy_by_station.png)", ""]

    # ---- 5. uncertainty
    L += ["## 5. Uncertainty: GeoCP intervals"]
    if g:
        sel = geo["selected"]
        bw = ("unweighted (plain split conformal)" if sel["bandwidth_m"] is None or np.isinf(float(sel["bandwidth_m"]))
              else f"{float(sel['bandwidth_m']) / 1000:g} km {sel['kernel']} kernel")
        L += [f"Each prediction carries an interval that should contain the observation {g['target_coverage']:.0%} of the time. "
              f"Weighting: **{bw}**, chosen by leave-one-station-out on the calibration stations "
              f"(coverage there {_pct(sel['loso_coverage'])}).", "",
              _table(["", "Value"],
                     [["Coverage on the test stations", f"{_pct(g['coverage'])} (target {g['target_coverage']:.0%})"],
                      ["Mean / median width", f"{_num(g['mean_width'])} / {_num(g['median_width'])} {UG}"],
                      ["Width relative to the prediction", _num(g["mean_width_to_prediction_ratio"])],
                      ["Coverage close to a calibration sensor", _pct(g.get("coverage_urban_core"))],
                      ["Coverage far from a calibration sensor", _pct(g.get("coverage_peripheral"))]]), "",
              "![coverage per station](plots/geocp_coverage.png)", ""]
    else:
        L += ["GeoCP has not been run for this snapshot.", ""]

    # ---- 6. spatial bias
    L += ["## 6. Are the errors spatially clustered?"]
    sb = summary.get("spatial_bias")
    if sb and "global_moran" in sb:
        gm = sb["global_moran"]
        L += [f"Moran's I of the out-of-fold residuals is **{_num(gm['I'], 3)}** (p = {_num(gm['p_sim'], 3)}, {sb.get('cells', '?')} grid cells). "
              + ("Significant: the model's errors cluster in space (see `plots/lisa_map.png`)." if gm.get("significant")
                 else "Not significant at the 5 % level: no evidence that errors cluster in space."), ""]
    else:
        L += [(sb or {}).get("skipped", "Not computed for this run."), ""]

    # ---- 7. training
    cv = info.get("cv", [])
    L += ["## 7. Training"]
    if cv:
        L += [_table(["Fold", "Held-out stations", f"RMSE ({UG})", f"MAE ({UG})", "R²", "Best epoch"],
                     [[c["fold"], len(c.get("val_stations") or []), _num(c["metrics"]["rmse"]), _num(c["metrics"]["mae"]),
                       _num(c["metrics"]["r2"], 3), f"{c['best_epoch']} of {c['epochs_run']}"] for c in cv], "lrrrrr"), ""]
    ff = info.get("final_fit")
    if ff:
        L += [f"Final model: {ff['epochs']} epoch(s) on all training stations ({ff['epochs_source']}). "
              "Training curves: ![loss](plots/loss_curve.png)", ""]

    # ---- 8. cautions
    L += ["## 8. Read with care"] + [f"- {n}" for n in _cautions(summary, info)] + [""]

    # ---- 9. files
    L += ["## 9. Files in this folder", "", _table(["File", "What it is"],
          [[f"`{f}`", describe(f)] for f in files], "ll"), ""]
    return "\n".join(L)
