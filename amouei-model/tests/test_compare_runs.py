import json

import numpy as np
import pandas as pd
import pytest

from reporting import compare_runs as cr

HOURS = pd.date_range("2025-03-01", periods=200, freq="h")


def _frame(stations=6, hours=HOURS, noise=1.0, seed=0, bias=0.0, offsets=None, fold_of=None):
    """Out-of-fold predictions of `stations` stations: truth = 20 + 5 sin(t); prediction = truth + bias + station offset + noise."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in range(1, stations + 1):
        truth = 20 + 5 * np.sin(np.arange(len(hours)) / 9.0 + s)
        shift = (offsets or {}).get(s, 0.0)
        parts.append(pd.DataFrame({"location_id": s, "timestamp": hours, "fold": (fold_of or {}).get(s, s % 3),
                                   "y_true": truth, "y_pred": truth + bias + shift + rng.normal(0, noise, len(hours))}))
    return pd.concat(parts, ignore_index=True)


def _run(root, arm, name, df, status="completed", seed=42, assignment="balanced", model=False, info=True):
    folder = root / arm / "checkpoints" / name
    folder.mkdir(parents=True)
    df.assign(timestamp=df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(folder / "oof_predictions.csv", index=False)
    if info:
        (folder / "run_info.json").write_text(json.dumps({"status": status, "hyperparameters": {"train": {"seed": seed}, "split": {"cv_assignment": assignment}}}))
    if model:
        (folder / "best_dual_branch_model.pt").write_bytes(b"x")
    return folder


def _arms(root):
    return cr.discover(root)


# ------------------------------------------------------------------------------ the numbers
def test_scores_are_the_exact_rmse_mae_r2_and_bias_of_the_common_rows(tmp_path):
    df = _frame(noise=0.0, bias=2.0)                                       # every prediction is 2 too high
    _run(tmp_path, "A", "r1", df)
    res = cr.compare(_arms(tmp_path), bootstrap=50)
    a = res["arms"].iloc[0]
    sst = ((df["y_true"] - df["y_true"].mean()) ** 2).sum()
    assert (a.rmse, a.mae, a.bias) == pytest.approx((2.0, 2.0, 2.0)) and a.r2 == pytest.approx(1 - 4.0 * len(df) / sst)
    assert res["checks"]["common_rows"] == len(df) and res["checks"]["stations"] == 6 and res["best"] == "A" and res["tied"] == []


def test_arms_are_ranked_and_the_spread_over_seeds_is_the_sample_sd(tmp_path):
    for i, seed in enumerate((1, 2)):
        _run(tmp_path, "good", f"r{i}", _frame(noise=1.0, seed=seed), seed=40 + i)
        _run(tmp_path, "bad", f"r{i}", _frame(noise=3.0, seed=seed + 10), seed=40 + i)
    res = cr.compare(_arms(tmp_path), bootstrap=500)
    arms = res["arms"].set_index("arm")
    runs = res["runs"].set_index("arm")
    assert res["best"] == "good" and list(res["arms"]["arm"]) == ["good", "bad"]
    assert arms.loc["good", "rmse"] == pytest.approx(runs.loc["good", "rmse"].mean())
    assert arms.loc["good", "rmse_sd"] == pytest.approx(runs.loc["good", "rmse"].std(ddof=1)) and arms.loc["good", "runs"] == 2
    assert arms.loc["bad", "gap_to_best"] > 1.5 and arms.loc["bad", "ci_low"] > 0 and not arms.loc["bad", "tied_with_best"]
    assert arms.loc["bad", "ci_low"] <= arms.loc["bad", "gap_to_best"] <= arms.loc["bad", "ci_high"]


def test_arms_of_the_same_quality_are_tied_and_the_preference_order_decides(tmp_path):
    for i in range(3):
        _run(tmp_path, "A", f"r{i}", _frame(noise=2.0, seed=i))
        _run(tmp_path, "B", f"r{i}", _frame(noise=2.0, seed=100 + i))
    res = cr.compare(_arms(tmp_path), bootstrap=500, prefer=["B", "A"])
    other = "B" if res["best"] == "A" else "A"
    assert res["tied"] == [other] and res["recommendation"] == "B"
    assert cr.compare(_arms(tmp_path), bootstrap=500)["recommendation"] == res["best"]               # no preference: the best


def test_a_clear_winner_is_not_overridden_by_the_preference_order(tmp_path):
    _run(tmp_path, "good", "r0", _frame(noise=1.0, seed=1))
    _run(tmp_path, "bad", "r0", _frame(noise=4.0, seed=2))
    res = cr.compare(_arms(tmp_path), bootstrap=500, prefer=["bad", "good"])
    assert res["tied"] == [] and res["recommendation"] == "good"


def test_the_bootstrap_resamples_whole_stations_so_few_stations_mean_a_wide_interval(tmp_path):
    """Arm B drifts differently at each station (offsets). With 3 stations the gap to A is poorly known, with 30 it is not."""
    widths = {}
    for n in (3, 30):
        root = tmp_path / str(n)
        rng = np.random.default_rng(5)
        offsets = {s: rng.normal(0, 2.0) for s in range(1, n + 1)}
        _run(root, "A", "r0", _frame(stations=n, noise=0.5, seed=1))
        _run(root, "B", "r0", _frame(stations=n, noise=0.5, seed=2, offsets=offsets))
        row = cr.compare(cr.discover(root), bootstrap=1000).get("arms").set_index("arm").loc["B"]
        widths[n] = row.ci_high - row.ci_low
    assert widths[3] > 2 * widths[30]


def test_each_bootstrap_draw_picks_exactly_as_many_stations_as_there_are_stations():
    w = cr.station_weights(7, 200, 3)
    assert w.shape == (200, 7) and (w.sum(axis=1) == 7).all() and (w >= 0).all()
    assert np.array_equal(w, cr.station_weights(7, 200, 3)) and not np.array_equal(w, cr.station_weights(7, 200, 4))
    assert 0 < (w == 0).mean() < 0.6 and w.max() >= 3                                      # some stations missed, some picked repeatedly


def test_a_clear_but_small_gap_inside_the_seed_to_seed_spread_counts_as_a_tie(tmp_path):
    """Seeds swing the RMSE by about 1; arm B is consistently 0.2 worse. The CI excludes 0, but nobody should pick on 0.2."""
    for i, b in enumerate((1.0, 2.0, 3.0)):
        _run(tmp_path, "A", f"r{i}", _frame(noise=0.0, bias=b))
        _run(tmp_path, "B", f"r{i}", _frame(noise=0.0, bias=b + 0.2))
    res = cr.compare(_arms(tmp_path), bootstrap=300)
    row = res["arms"].set_index("arm").loc["B"]
    assert res["best"] == "A" and row.gap_to_best == pytest.approx(0.2) and row.ci_low > 0 and row.rmse_sd == pytest.approx(1.0)
    assert row.tied_with_best and res["tied"] == ["B"]


def test_the_result_is_reproducible_for_a_seed(tmp_path):
    _run(tmp_path, "A", "r0", _frame(seed=1))
    _run(tmp_path, "B", "r0", _frame(seed=2, noise=2.0))
    a, b = (cr.compare(_arms(tmp_path), bootstrap=300, seed=9)["arms"] for _ in range(2))
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------------------ which rows count
def test_only_the_rows_every_run_has_are_scored(tmp_path):
    full = _frame(noise=0.0, bias=1.0)
    half = _frame(hours=HOURS[:100], noise=0.0, bias=1.0)
    _run(tmp_path, "long", "r0", full)
    _run(tmp_path, "short", "r0", half)
    res = cr.compare(_arms(tmp_path), bootstrap=50)
    assert res["checks"]["common_rows"] == 6 * 100
    arms = res["arms"].set_index("arm")
    assert arms.loc["long", "common_share_of_own_rows"] == pytest.approx(0.5) and arms.loc["short", "common_share_of_own_rows"] == pytest.approx(1.0)
    assert arms.loc["long", "rmse"] == pytest.approx(1.0) == arms.loc["short", "rmse"]


def test_runs_with_nothing_in_common_are_refused(tmp_path):
    _run(tmp_path, "A", "r0", _frame(hours=HOURS[:50]))
    _run(tmp_path, "B", "r0", _frame(hours=HOURS[100:150]))
    with pytest.raises(ValueError, match="share no"):
        cr.compare(_arms(tmp_path))


def test_rows_where_the_runs_disagree_about_the_truth_are_dropped_and_counted(tmp_path):
    a = _frame(noise=0.0, bias=1.0)
    b = a.copy()
    changed = (b["location_id"] == 1) & (b.index < 10)
    b.loc[changed, "y_true"] += 5.0                                         # e.g. a spike that one dataset removed and interpolated
    _run(tmp_path, "A", "r0", a)
    _run(tmp_path, "B", "r0", b)
    res = cr.compare(_arms(tmp_path), bootstrap=50)
    assert res["checks"]["rows_dropped_for_truth_disagreement"] == int(changed.sum()) and res["checks"]["common_rows"] == len(a) - int(changed.sum())


def test_a_run_that_lists_a_station_hour_twice_is_refused(tmp_path):
    df = _frame()
    _run(tmp_path, "A", "r0", pd.concat([df, df.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="more than once"):
        cr.compare(_arms(tmp_path))


# ------------------------------------------------------------------------------ unfinished runs, folds, balance
def test_unfinished_runs_are_skipped_unless_asked_for(tmp_path):
    _run(tmp_path, "A", "done", _frame(seed=1))
    _run(tmp_path, "A", "killed", _frame(seed=2), status="running")
    _run(tmp_path, "B", "old_but_trained", _frame(seed=3), info=False, model=True)           # an older run: no status, but it has its model
    _run(tmp_path, "B", "old_unfinished", _frame(seed=4), info=False)
    res = cr.compare(_arms(tmp_path), bootstrap=50)
    assert res["checks"]["runs"] == 2 and {(s["arm"], s["run"]) for s in res["skipped"]} == {("A", "killed"), ("B", "old_unfinished")}
    assert cr.compare(_arms(tmp_path), include_incomplete=True, bootstrap=50)["checks"]["runs"] == 4
    assert "killed (running)" in cr.to_markdown(res)


def test_the_tool_says_whether_every_run_used_the_same_folds(tmp_path):
    same = {s: s % 3 for s in range(1, 7)}
    _run(tmp_path, "A", "r0", _frame(seed=1, fold_of=same), assignment="by_station")
    _run(tmp_path, "B", "r0", _frame(seed=2, fold_of=same), assignment="by_station")
    res = cr.compare(_arms(tmp_path), bootstrap=50)
    assert res["checks"]["stations_in_different_folds_across_runs"] == 0 and res["checks"]["cv_assignment"] == ["by_station"]
    assert "Same CV folds for every station in every run: **yes**" in cr.to_markdown(res)

    moved = {**same, 3: (same[3] + 1) % 3, 5: (same[5] + 2) % 3}
    _run(tmp_path, "C", "r0", _frame(seed=3, fold_of=moved))
    res = cr.compare(_arms(tmp_path), bootstrap=50)
    assert res["checks"]["stations_in_different_folds_across_runs"] == 2 and res["checks"]["cv_assignment"] == ["balanced", "by_station"]
    md = cr.to_markdown(res)
    assert "**NO** (2 of 6 stations change fold" in md and "by_station" in md


def test_the_weight_of_the_biggest_stations_is_reported(tmp_path):
    df = _frame(stations=4)
    df = df[~((df["location_id"] > 1) & (df.groupby("location_id").cumcount() >= 20))]     # station 1: 200 rows, the others: 20
    _run(tmp_path, "A", "r0", df)
    c = cr.compare(_arms(tmp_path), bootstrap=50)["checks"]
    assert c["largest_station_share"] == pytest.approx(200 / 260) and c["smallest_station_rows"] == 20 and c["top5_station_share"] == pytest.approx(1.0)


def test_one_arm_with_one_run_still_works(tmp_path):
    _run(tmp_path, "only", "r0", _frame())
    res = cr.compare(_arms(tmp_path), bootstrap=50)
    a = res["arms"].iloc[0]
    assert res["best"] == res["recommendation"] == "only" and a.gap_to_best == 0 and np.isnan(a.rmse_sd) and not a.tied_with_best
    assert "Best by RMSE: `only`" in cr.to_markdown(res)


def test_no_completed_runs_is_an_error(tmp_path):
    _run(tmp_path, "A", "r0", _frame(), status="running")
    with pytest.raises(ValueError, match="no completed runs"):
        cr.compare(_arms(tmp_path))


# ------------------------------------------------------------------------------ finding runs, the command line
def test_runs_are_found_by_folder_layout(tmp_path):
    f = _run(tmp_path, "A", "r0", _frame())
    _run(tmp_path, "B", "r0", _frame())
    (tmp_path / "C" / "checkpoints").mkdir(parents=True)                                       # an arm without runs
    assert cr.run_folders(f) == [f] and cr.run_folders(tmp_path / "A" / "checkpoints") == [f] and cr.run_folders(tmp_path / "nothing") == []
    assert set(cr.discover(tmp_path)) == {"A", "B"}


def test_the_command_line_writes_the_report_and_the_tables(tmp_path, monkeypatch, capsys):
    _run(tmp_path, "good", "r0", _frame(noise=1.0, seed=1))
    _run(tmp_path, "bad", "r0", _frame(noise=4.0, seed=2))
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["compare_runs", "--runs-root", str(tmp_path), "--out-dir", str(out), "--bootstrap", "100", "--prefer", "bad,good"])
    cr.main()
    assert "Best by RMSE: `good`" in capsys.readouterr().out
    assert (out / "comparison.md").is_file() and pd.read_csv(out / "comparison_arms.csv")["arm"].tolist() == ["good", "bad"]
    assert len(pd.read_csv(out / "comparison_runs.csv")) == 2


def test_explicit_arms_can_be_named_and_a_missing_path_is_a_clear_error(tmp_path, monkeypatch, capsys):
    a = _run(tmp_path / "x", "q", "r0", _frame(seed=1))
    b = _run(tmp_path / "y", "q", "r0", _frame(seed=2, noise=3.0))
    monkeypatch.setattr("sys.argv", ["c", "--arm", f"first={a}", "--arm", f"second={b.parent}", "--out-dir", str(tmp_path / "o"), "--bootstrap", "50"])
    cr.main()
    assert "`first`" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["c", "--arm", f"nope={tmp_path / 'missing'}"])
    with pytest.raises(SystemExit, match="oof_predictions.csv"):
        cr.main()
