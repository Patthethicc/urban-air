import json

from networks.dual_branch_net import DualBranchNet
from training.run_info import load_run_info, new_run_info, save_run_info, update_run_info, update_runs_index


def test_run_info_records_the_hyperparameters_and_updates_in_place(cfg, tmp_path):
    model = DualBranchNet(cfg["model"])
    info = new_run_info("20260101_000000", cfg, model, {"rows_per_partition": {"train": 1}})
    save_run_info(tmp_path, info)

    loaded = load_run_info(tmp_path)
    assert loaded["hyperparameters"]["model"]["temporal_lstm"]["hidden_size"] == cfg["model"]["temporal_lstm"]["hidden_size"]
    assert loaded["parameters"]["total"] == model.parameter_counts()["total"]
    assert "DualBranchNet" in loaded["architecture"]

    update_run_info(tmp_path, test_metrics={"rmse": 1.5})
    assert load_run_info(tmp_path)["test_metrics"]["rmse"] == 1.5
    json.loads((tmp_path / "run_info.json").read_text())             # valid JSON (no bare Infinity)


def test_runs_index_has_one_row_per_run(cfg, tmp_path):
    model = DualBranchNet(cfg["model"])
    for rid in ("a", "b", "a"):
        update_runs_index(tmp_path, new_run_info(rid, cfg, model, {}))
    rows = (tmp_path / "runs_index.csv").read_text().strip().splitlines()
    assert len(rows) == 3                                           # header + runs a and b
