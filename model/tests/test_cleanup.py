import os
from pathlib import Path

import pytest

from cleanup import execute_cleanup, human, plan_cleanup


def _cfg(root: Path) -> dict:
    return {"data": {"paths": {"processed_dir": str(root / "processed"), "checkpoints_dir": str(root / "checkpoints"),
                               "snapshots_dir": str(root / "snapshots"), "tuning_dir": str(root / "tuning"),
                               "raster_work_dir": str(root / "processed" / "rasters")}},
            "output": {"basemap_cache_dir": str(root / ".tile_cache")}}


def _write(path: Path, text: str = "x" * 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def project(tmp_path):
    """A fake model/ folder: generated outputs plus things a reset must never touch."""
    root = tmp_path / "model"
    for folder in ("processed", "checkpoints", "snapshots", "tuning"):
        _write(root / folder / ".gitkeep", "")
    _write(root / "processed" / "patches.npy")
    _write(root / "processed" / "rasters" / "master_morphology_stack_10m.tif", "r" * 5000)      # slow to rebuild
    _write(root / "checkpoints" / "20260101_000000" / "best_dual_branch_model.pt", "m" * 300)
    _write(root / "checkpoints" / "runs_index.csv")
    _write(root / "snapshots" / "20260101_000000" / "metrics_summary.json")
    _write(root / "tuning" / "20260101_000001" / "best_hyperparameters.json")
    _write(root / ".tile_cache" / "tile.png")
    for bystander in ("configs/data.yaml", ".env", "README.md", "run_pipeline.py"):               # never deleted
        _write(root / bystander, "keep me")
    _write(tmp_path / "pilot testing" / "Air Data" / "sensor.csv", "input data")                  # input data, outside model/
    return root


def _left(root: Path) -> set[str]:
    return {str(p.relative_to(root)).replace(os.sep, "/") for p in root.rglob("*") if p.is_file()}


def test_a_dry_run_lists_what_would_go_and_deletes_nothing(project):
    before = _left(project)
    plan = plan_cleanup(_cfg(project), root=project)
    assert [t.name for t in plan] == ["processed", "checkpoints", "snapshots", "tuning"]
    assert sum(t.files for t in plan) == 5                          # patches, model, runs_index, metrics, best_hp
    assert _left(project) == before


def test_the_counts_and_sizes_are_right(project):
    plan = {t.name: t for t in plan_cleanup(_cfg(project), root=project)}
    assert plan["processed"].files == 1 and plan["processed"].bytes == 100          # patches.npy only: rasters are kept
    assert plan["checkpoints"].files == 2 and plan["checkpoints"].bytes == 400
    assert plan["snapshots"].files == 1 and plan["tuning"].files == 1


def test_executing_removes_the_outputs_but_keeps_the_folders_and_their_placeholders(project):
    files, nbytes = execute_cleanup(plan_cleanup(_cfg(project), root=project))
    assert files == 5 and nbytes == 100 + 400 + 100 + 100
    for folder in ("processed", "checkpoints", "snapshots", "tuning"):
        assert (project / folder / ".gitkeep").is_file(), folder
    assert not (project / "checkpoints" / "20260101_000000").exists() and not (project / "processed" / "patches.npy").exists()


def test_the_built_master_raster_and_the_tile_cache_are_kept_by_default(project):
    execute_cleanup(plan_cleanup(_cfg(project), root=project))
    assert (project / "processed" / "rasters" / "master_morphology_stack_10m.tif").is_file()
    assert (project / ".tile_cache" / "tile.png").is_file()


def test_the_dry_run_says_the_rasters_are_kept(project):
    plan = {t.name: t for t in plan_cleanup(_cfg(project), root=project)}
    assert [k.name for k in plan["processed"].kept] == ["rasters"]


def test_include_rasters_and_include_tile_cache_remove_them_too(project):
    plan = plan_cleanup(_cfg(project), include_rasters=True, include_tile_cache=True, root=project)
    assert plan[-1].name == "tile_cache"
    execute_cleanup(plan)
    assert not (project / "processed" / "rasters").exists() and not (project / ".tile_cache" / "tile.png").exists()


def test_keep_leaves_a_whole_folder_alone(project):
    execute_cleanup(plan_cleanup(_cfg(project), keep=("tuning",), root=project))
    assert (project / "tuning" / "20260101_000001" / "best_hyperparameters.json").is_file()
    assert not (project / "snapshots" / "20260101_000000").exists()


def test_an_unknown_keep_name_is_rejected(project):
    with pytest.raises(ValueError, match="--keep"):
        plan_cleanup(_cfg(project), keep=("configs",), root=project)


def test_configs_env_source_and_input_data_are_never_touched(project):
    execute_cleanup(plan_cleanup(_cfg(project), include_rasters=True, include_tile_cache=True, root=project))
    for survivor in ("configs/data.yaml", ".env", "README.md", "run_pipeline.py"):
        assert (project / survivor).read_text() == "keep me", survivor
    assert (project.parent / "pilot testing" / "Air Data" / "sensor.csv").read_text() == "input data"


@pytest.mark.parametrize("bad", ["../pilot testing", "..", ".", "/", "configs"])
def test_a_folder_outside_the_model_dir_or_the_model_dir_itself_is_refused(project, bad):
    cfg = _cfg(project)
    cfg["data"]["paths"]["processed_dir"] = str((project / bad).resolve()) if bad != "/" else os.path.abspath(os.sep)
    if bad == "configs":                                    # inside model/, so allowed: it is just a (wrong) folder name
        plan_cleanup(cfg, root=project)
        return
    with pytest.raises(ValueError, match="refusing"):
        plan_cleanup(cfg, root=project)


def test_a_misconfigured_cache_folder_outside_the_model_dir_is_refused(project, tmp_path):
    cfg = _cfg(project)
    cfg["output"]["basemap_cache_dir"] = str(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="refusing"):
        plan_cleanup(cfg, include_tile_cache=True, root=project)


def test_symbolic_links_are_removed_never_followed(project, tmp_path):
    outside = tmp_path / "precious"
    _write(outside / "important.txt", "do not delete")
    try:
        os.symlink(outside, project / "snapshots" / "link_to_elsewhere", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")
    execute_cleanup(plan_cleanup(_cfg(project), root=project))
    assert (outside / "important.txt").read_text() == "do not delete"
    assert not (project / "snapshots" / "link_to_elsewhere").exists()


def test_missing_folders_are_fine_and_get_their_placeholder_back(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    execute_cleanup(plan_cleanup(_cfg(root), root=root))
    assert (root / "processed" / ".gitkeep").is_file()


def test_sizes_are_printed_readably():
    assert human(512) == "512 B" and human(2048) == "2.0 KB" and human(5 * 1024 ** 2) == "5.0 MB"
