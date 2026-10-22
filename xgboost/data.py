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

def _load_dotenv(path):
    """Read KEY=VALUE lines from `path` into os.environ, WITHOUT overwriting anything already set.

    A real environment variable always wins, so `$env:XGB_FOLDS='pilot'` still overrides the file for
    one command. Deliberately ~15 lines instead of a python-dotenv dependency: nothing else in this
    branch needs it, and this keeps the install list unchanged. Blank lines, `#` comments and an
    optional `export ` prefix are ignored, as are quotes around the value.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            if line.startswith('export '):
                line = line[7:]
            key, _, val = line.partition('=')
            key, val = key.strip(), val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val


# Load XGBoost/.env once, before any switch below is read. This is what lets the split be chosen from a
# file instead of a per-session PowerShell variable.
_load_dotenv(os.path.join(ROOT, '.env'))


# ---------------------------------------------------------------------------------------------
# Which DATASET the pipeline runs on. Set the environment variable to switch:
#
#     XGB_DATASET=full    (default) 84 locations, 1.49 M rows, 3,696-cell grid  <- the real data
#     XGB_DATASET=pilot             18 sensors,    43 k rows,   378-cell grid    <- the pilot record
#
# The pilot files are never overwritten, so the published pilot results stay reproducible: running
# with XGB_DATASET=pilot restores exactly the 378-cell / 11-3-3 configuration that produced them.
# ---------------------------------------------------------------------------------------------
DATASET = os.environ.get('XGB_DATASET', 'full').strip().lower()
if DATASET not in ('full', 'pilot'):
    raise ValueError(f"XGB_DATASET must be 'full' or 'pilot', got {DATASET!r}")

_FILES = {
    'full': {'master': 'Master_DL_Ready_full.csv', 'grid': 'morphology_grid_full.csv',
             'folds': 'fold_partitions_full.json', 'mapping': 'sensor_to_grid_map_full.csv'},
    'pilot': {'master': 'Master_DL_Ready.csv', 'grid': 'full_morphology_grid.csv',
              'folds': 'fold_partitions.json', 'mapping': 'sensor_to_grid_map_full.csv'},
}[DATASET]

# ---- inputs ---------------------------------------------------------------------------------
MASTER_CSV = os.path.join(DATA_DIR, _FILES['master'])
GRID_CSV = os.path.join(DATA_DIR, _FILES['grid'])
GRID_WKT_CSV = os.path.join(DATA_DIR, 'full_morphology_grid_wkt.csv')
SENSOR_MAP_CSV = os.path.join(DATA_DIR, _FILES['mapping'])
# XGB_FOLDS=shared switches every stage to the split shared with the CNN-LSTM run, so one command can
# reproduce the comparable experiment without editing files:
#     $env:XGB_FOLDS='shared'; python run_pipeline.py --stage all
_FOLDS_OVERRIDE = os.environ.get('XGB_FOLDS', '').strip().lower()
# The shared split is defined on the full 83-station network only: it names stations the 18-sensor
# pilot does not contain, so honouring it there would leave every partition empty (which made
# checks/verify_geocp.py divide by zero). Ignore it for any other dataset, with a loud note.
if _FOLDS_OVERRIDE == 'shared' and DATASET != 'full':
    print('[data] XGB_FOLDS=shared ignored: the shared split exists only for the full network '
          '(dataset=%s), using this dataset\'s own folds' % DATASET)
    _FOLDS_OVERRIDE = ''
FOLD_JSON = (os.path.join(DATA_DIR, 'fold_partitions_shared.json')
             if _FOLDS_OVERRIDE == 'shared' else os.path.join(DATA_DIR, _FILES['folds']))
LOCATIONS_CSV = os.path.join(DATA_DIR, 'sensor_locations.csv')
RASTER_TIF = os.path.join(DATA_DIR, 'master_raster_v2.tif')

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
