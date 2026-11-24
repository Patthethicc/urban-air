"""Reset: delete everything the pipeline GENERATED so the next run starts from nothing.

Meant for "I only tried the pilot data, start over with the real data". It removes
    processed/    the arrays the model trains on (patches, windows, scalers, partition)
    checkpoints/  trained models, their settings, run comparison table
    snapshots/    metrics, intervals, plots, maps, rasters written by geocp / report / infer
    tuning/       hyperparameter search results (tuned on the data you are discarding)
and never touches configs, .env, source code or your INPUT data (paths outside model/ are refused).

Two things are kept unless you ask: the Master GeoTIFF built by `--stage rasters` (processed/rasters/, slow to
rebuild) and the web-tile cache. Folder placeholders (.gitkeep) stay. Without --yes nothing is deleted.
"""
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

SCOPES = ("processed", "checkpoints", "snapshots", "tuning")
MODEL_DIR = Path(__file__).parent


@dataclass
class Target:
    name: str
    path: Path
    entries: list = field(default_factory=list)       # top-level children that will be deleted
    kept: list = field(default_factory=list)          # top-level children that are protected
    files: int = 0
    bytes: int = 0


def _inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path != root and root in path.parents


def _size(path: Path) -> tuple[int, int]:
    if path.is_symlink() or path.is_file():
        return 1, 0 if path.is_symlink() else path.stat().st_size
    files = nbytes = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            files += 1
            nbytes += p.stat().st_size
    return files, nbytes


def human(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def plan_cleanup(cfg: dict, keep=(), include_rasters: bool = False, include_tile_cache: bool = False,
                 root: Path = MODEL_DIR) -> list[Target]:
    """What a reset would delete. Raises ValueError if a configured folder is outside `root` (the model/ folder)."""
    from config_parser import resolve_path

    paths = cfg["data"]["paths"]
    folders = {"processed": paths["processed_dir"], "checkpoints": paths["checkpoints_dir"],
               "snapshots": paths["snapshots_dir"], "tuning": paths["tuning_dir"]}
    unknown = set(keep) - set(SCOPES)
    if unknown:
        raise ValueError(f"--keep takes {SCOPES}, got {sorted(unknown)}")
    rasters = Path(paths["raster_work_dir"]).resolve()

    plan = []
    for name, folder in folders.items():
        if name in keep:
            continue
        path = Path(folder)
        if not _inside(path, root):
            raise ValueError(f"refusing to clean {path}: it is not inside {Path(root).resolve()} "
                             f"(check data.paths.{name}_dir; input data is never deleted)")
        target = Target(name, path)
        for child in sorted(path.iterdir()) if path.is_dir() else []:
            if child.name == ".gitkeep":
                continue
            real = child.resolve()
            protects_rasters = not include_rasters and (real == rasters or real in rasters.parents)
            if protects_rasters:
                target.kept.append(child)
                continue
            files, nbytes = _size(child)
            target.entries.append(child)
            target.files += files
            target.bytes += nbytes
        plan.append(target)

    if include_tile_cache:
        cache = Path(resolve_path(cfg["output"]["basemap_cache_dir"]))
        if not _inside(cache, root):
            raise ValueError(f"refusing to clean {cache}: it is not inside {Path(root).resolve()}")
        target = Target("tile_cache", cache)
        for child in sorted(cache.iterdir()) if cache.is_dir() else []:
            files, nbytes = _size(child)
            target.entries.append(child)
            target.files += files
            target.bytes += nbytes
        plan.append(target)
    return plan


def execute_cleanup(plan: list[Target]) -> tuple[int, int]:
    """Delete what `plan_cleanup` listed. Returns (files, bytes) removed. Symbolic links are removed, never followed."""
    files = nbytes = 0
    for target in plan:
        for child in target.entries:
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
        files, nbytes = files + target.files, nbytes + target.bytes
        if target.name in SCOPES:                               # keep the tracked placeholder
            target.path.mkdir(parents=True, exist_ok=True)
            (target.path / ".gitkeep").touch()
    return files, nbytes


def describe(plan: list[Target], root: Path = MODEL_DIR) -> str:
    rows = []
    for t in plan:
        rel = os.path.relpath(t.path, root)
        note = f"   (kept: {', '.join(os.path.relpath(k, root) for k in t.kept)})" if t.kept else ""
        rows.append(f"  {rel + '/':<16}{t.files:>6} files{human(t.bytes):>12}{note}")
    return "\n".join(rows)
