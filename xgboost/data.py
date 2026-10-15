"""Where the inputs, the split and the results live, resolved from the project root.

Why this module exists: every script used to open paths like 'Master_DL_Ready.csv' relative to the
current working directory, so running anything from a subfolder (or from the repository root) silently
looked in the wrong place -- and `sensor_to_grid_map_full.csv` is WRITTEN at import time, so a wrong
cwd quietly created a stray file. The CNN-LSTM branch solved the same problem with its
`config_parser.py`. This is the equivalent: one place that knows the layout, and `DATA_DIR`,
`ROOT` and `OUT_DIR` are absolute as a result.

Layout (data at the project root, code in subpackages):

    XGBoost/
      data/         Master_DL_Ready.csv, full_morphology_grid.csv, fold_partitions.json, ...
      outputs/      results
      configs/      settings
      <packages>/   code
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(ROOT, 'outputs')
CONFIG_DIR = os.path.join(ROOT, 'configs')

# ---- inputs ---------------------------------------------------------------------------------
MASTER_CSV = os.path.join(DATA_DIR, 'Master_DL_Ready.csv')
GRID_CSV = os.path.join(DATA_DIR, 'full_morphology_grid.csv')
GRID_WKT_CSV = os.path.join(DATA_DIR, 'full_morphology_grid_wkt.csv')
SENSOR_MAP_CSV = os.path.join(DATA_DIR, 'sensor_to_grid_map_full.csv')
FOLD_JSON = os.path.join(DATA_DIR, 'fold_partitions.json')

# ---- outputs --------------------------------------------------------------------------------
ABLATION_RESULTS = os.path.join(OUT_DIR, 'ablation_results.csv')
ABLATION_SUMMARY = os.path.join(OUT_DIR, 'ablation_summary.csv')
ABLATION_PARAMS = os.path.join(OUT_DIR, 'ablation_params.json')
GEOCP_DIR = os.path.join(OUT_DIR, 'geocp')


# Every file that may be referenced by bare filename, so `resolve` can map it into data/.
KNOWN = {
    os.path.basename(MASTER_CSV): MASTER_CSV,
    os.path.basename(GRID_CSV): GRID_CSV,
    os.path.basename(GRID_WKT_CSV): GRID_WKT_CSV,
    os.path.basename(SENSOR_MAP_CSV): SENSOR_MAP_CSV,
    os.path.basename(FOLD_JSON): FOLD_JSON,
}


def resolve(path, default=None):
    """Return an absolute path for a data file.

    Accepts an absolute path (returned as-is), a path that already exists relative to the working
    directory, a bare filename (looked up in data/), or None (falls back to `default`, then to the
    known-file table).

    The bare-filename case exists because the notebook and some call sites still pass names like
    'sensor_to_grid_map_full.csv' to functions whose default used to be a cwd-relative literal.
    Resolving them here means the file is found whether or not the working directory is XGBoost/,
    instead of raising FileNotFoundError or -- worse, for the file this pipeline writes -- silently
    creating a stray copy beside the cwd.
    """
    if path is None:
        if default is None:
            return None
        path = default
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    return KNOWN.get(base, os.path.join(DATA_DIR, base))


def out(*parts):
    """Path inside outputs/, creating the parent directories if needed."""
    p = os.path.join(OUT_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def ensure_dirs():
    """Create the folders the pipeline writes into. Called at import of the prep module."""
    for d in (DATA_DIR, OUT_DIR, GEOCP_DIR):
        os.makedirs(d, exist_ok=True)
