"""inference.openaq_only_from_year: from that year on, the stand-in history of a cell comes from the OpenAQ reference
stations only. The Quezon City sensors (ids from 9,000,000) have no readings in 2026, so without it the cells whose nearest
stations are all QC stations get no prediction."""
import numpy as np
import pandas as pd
import pytest
import torch
from rasterio.transform import from_origin

from config_parser import apply_overrides, validate
from inference.cell_inference import openaq_only, padded_raster, predict_surface, reference_stations
from preprocessing.qc_convert import ID_BASE
from preprocessing.scalers import Scalers

OPENAQ, QC = 1, 9_000_001
START = pd.Timestamp("2025-12-31 00:00")
BEFORE, AFTER = pd.Timestamp("2025-12-31 12:00"), pd.Timestamp("2026-01-01 12:00")        # hour 12 and hour 36 of the matrix


def _centre(r, c):
    return c * 10 + 5.0, 95.0 - r * 10


class _LastLag(torch.nn.Module):
    """Ignores the patch and returns the (scaled) PM2.5 of the most recent lag step."""
    def forward(self, spatial, temporal):
        return temporal[:, -1, :1]


@pytest.fixture
def world(cfg, tmp_path):
    """A 10 x 10 raster with a QC station in its bottom-right cell and an OpenAQ station in its top-left cell (the QC one
    listed first, so the OpenAQ-only subset is NOT a prefix of the reference list); k = 1, so a cell takes the reading of
    whichever station is nearest. OpenAQ reads 10 + hour, QC reads 100 + hour."""
    import rasterio
    path = tmp_path / "r.tif"
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32", crs="EPSG:32651",
                       transform=from_origin(0, 100, 10, 10)) as dst:
        dst.write(np.ones((1, 10, 10), "float32"))
    cfg["data"]["spatial_lag"]["k"] = 1
    cfg["derived"]["patch_px"] = 1
    cfg["inference"].update(stride_px=1, chunk_cells=30, window=None, progress=False, openaq_only_from_year=2026)
    hours = np.arange(72, dtype=float)
    values = np.column_stack([100 + hours, 10 + hours])                           # columns: QC, OpenAQ
    padded, transform, _, shape = padded_raster(str(path), 1)
    scalers = Scalers().fit(np.ones((2, 1, 1, 1)), np.array([[10.0], [30.0]]), [0, 1], [])        # mean 20, std 10
    return {"cfg": cfg, "values": values, "padded": padded, "transform": transform, "shape": shape, "scalers": scalers,
            "ref_xy": np.array([_centre(9, 9), _centre(0, 0)])}


def _run(w, stamps, values=None, ids=(QC, OPENAQ)):
    """`ids` name the two stations (columns of the hourly matrix, which are also the reference stations)."""
    hourly = {"values": w["values"] if values is None else values, "location_ids": np.array(ids), "start": START}
    return predict_surface(_LastLag(), w["scalers"], w["padded"], w["shape"], w["transform"], hourly, list(ids), w["ref_xy"],
                           stamps, w["cfg"])[0]


# ------------------------------------------------------------------------------ the rule
def test_the_year_of_the_predicted_hour_switches_it_on_and_null_switches_it_off(cfg):
    cfg["inference"]["openaq_only_from_year"] = 2026
    assert not openaq_only("2025-12-31 23:00", cfg) and openaq_only("2026-01-01 00:00", cfg) and openaq_only("2027-06-01", cfg)
    cfg["inference"]["openaq_only_from_year"] = None
    assert not openaq_only("2026-01-01 00:00", cfg)
    del cfg["inference"]["openaq_only_from_year"]                                  # a config written before the option existed
    assert not openaq_only("2026-01-01 00:00", cfg)


def test_the_option_keeps_only_the_openaq_reference_stations_and_their_coordinates(cfg):
    ids, xy = reference_stations([QC, OPENAQ, 7, QC + 1], np.array([[0, 0], [1, 1], [2, 2], [3, 3]]), True, cfg)
    assert ids == [OPENAQ, 7] and xy.tolist() == [[1, 1], [2, 2]]
    assert reference_stations([ID_BASE - 1, ID_BASE], np.array([[0, 0], [1, 1]]), True, cfg)[0] == [ID_BASE - 1]   # QC starts at ID_BASE
    ids, xy = reference_stations([QC, OPENAQ], np.array([[0, 0], [1, 1]]), False, cfg)
    assert ids == [QC, OPENAQ] and xy.shape == (2, 2)                              # off: untouched
    with pytest.raises(ValueError, match="asks for OpenAQ stations only"):
        reference_stations([QC, QC + 1], np.zeros((2, 2)), True, cfg)


def test_the_setting_is_validated(cfg):
    for ok in ("2026", "null", "2031"):
        validate(apply_overrides(cfg, [f"inference.openaq_only_from_year={ok}"]))
    assert cfg["inference"]["openaq_only_from_year"] == 2031
    for bad in ("true", "'2026'", "26", "2026.5", "1999"):
        with pytest.raises(ValueError, match="openaq_only_from_year"):
            validate(apply_overrides(cfg, [f"inference.openaq_only_from_year={bad}"]))


# ------------------------------------------------------------------------------ the surfaces
def test_a_2025_hour_uses_every_station_and_a_2026_hour_only_the_openaq_ones(world):
    pred = _run(world, [BEFORE, AFTER])
    assert pred.shape == (2, 10, 10)
    # the cell of the QC station: its own sensor in 2025 (100 + hour 11), the nearest OpenAQ station in 2026 (10 + hour 35)
    assert pred[0, 9, 9] == pytest.approx(111.0, rel=1e-4) and pred[1, 9, 9] == pytest.approx(45.0, rel=1e-4)
    assert pred[0, 0, 0] == pytest.approx(21.0, rel=1e-4) and pred[1, 0, 0] == pytest.approx(45.0, rel=1e-4)
    assert np.isfinite(pred).all()
    # each hour on its own gives the same surface as inside the mixed stack (the neighbour sets do not leak between hours)
    assert np.allclose(_run(world, BEFORE), pred[0], equal_nan=True) and np.allclose(_run(world, AFTER), pred[1], equal_nan=True)


def test_switched_off_a_2026_hour_still_reads_the_qc_station(world):
    world["cfg"]["inference"]["openaq_only_from_year"] = None
    assert _run(world, AFTER)[9, 9] == pytest.approx(135.0, rel=1e-4)              # 100 + hour 35


def test_the_cells_the_silent_qc_network_would_leave_empty_get_a_prediction(world):
    values = world["values"].copy()
    values[24:, 0] = np.nan                                                        # the QC station (column 0) is silent from 2026-01-01
    off = {**world["cfg"], "inference": {**world["cfg"]["inference"], "openaq_only_from_year": None}}
    holes = _run({**world, "cfg": off}, AFTER, values=values)
    assert np.isnan(holes[5:, 5:]).any() and np.isfinite(holes[0, 0])              # the cells nearest the QC station: no prediction
    filled = _run(world, AFTER, values=values)
    assert np.isfinite(filled).all() and filled[9, 9] == pytest.approx(45.0, rel=1e-4)
    assert np.array_equal(filled[np.isfinite(holes)], holes[np.isfinite(holes)])   # cells that already had one are unchanged


def test_a_2026_date_with_no_openaq_reference_station_is_refused_before_any_work(world):
    with pytest.raises(ValueError, match="asks for OpenAQ stations only"):
        _run(world, AFTER, ids=(QC, QC + 1))
    assert np.isfinite(_run(world, BEFORE, ids=(QC, QC + 1))).all()            # but 2025 does not need an OpenAQ station
