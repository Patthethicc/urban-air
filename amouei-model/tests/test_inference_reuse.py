"""predict_surface computes each chunk's CNN embedding once and reuses it for every hour: the result must equal one
plain forward pass per hour, for every model shape, every validity pattern and every chunking."""
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from inference import cell_inference
from inference.cell_inference import padded_raster, predict_surface
from networks.dual_branch_net import DualBranchNet
from preprocessing.scalers import Scalers

ROWS, COLS, RES, PX = 30, 40, 10.0, 8
START = pd.Timestamp("2025-01-01 00:00")
HOURS = [START + pd.Timedelta(hours=h) for h in (6, 7, 8, 9)]


def _centre(r, c):
    return c * RES + RES / 2, ROWS * RES - (r * RES + RES / 2)


@pytest.fixture
def world(cfg, tmp_path):
    """A 30 x 40 raster (1,200 cells = 3 chunks of 500, the last one short) and two stations: A near the top rows,
    B near the bottom rows. With k = 1 every cell uses its nearest station only."""
    path = tmp_path / "r.tif"
    rng = np.random.default_rng(0)
    with rasterio.open(path, "w", driver="GTiff", height=ROWS, width=COLS, count=5, dtype="float32", crs="EPSG:32651",
                       transform=from_origin(0, ROWS * RES, RES, RES)) as dst:
        dst.write(rng.uniform(0, 1, (5, ROWS, COLS)).astype("float32"))
    cfg["data"]["spatial_lag"]["k"] = 1
    cfg["derived"]["patch_px"] = PX
    cfg["inference"].update(stride_px=1, chunk_cells=500, window=None, progress=False)
    hours = 24
    values = np.column_stack([20 + 3 * np.sin(np.arange(hours)), 35 + 2 * np.cos(np.arange(hours))]).astype(float)
    padded, transform, _, shape = padded_raster(str(path), PX)
    scalers = Scalers().fit(rng.uniform(0, 1, (6, 5, PX, PX)), np.array([[10.0], [20.0], [30.0], [25.0], [15.0], [40.0]]),
                            [0, 1], [])
    return {"cfg": cfg, "values": values, "padded": padded, "transform": transform, "shape": shape, "scalers": scalers,
            "ref_xy": np.array([_centre(2, 20), _centre(27, 20)])}


def _model(cfg, spatial=True):
    torch.manual_seed(0)
    mc = {**cfg["model"], "spatial_cnn": {**cfg["model"]["spatial_cnn"], "enabled": spatial, "patch_px": PX}}
    return DualBranchNet(mc)


def _run(w, model, stamps, values=None, **kw):
    hourly = {"values": w["values"] if values is None else values, "location_ids": np.array([1, 2]), "start": START}
    return predict_surface(model, w["scalers"], w["padded"], w["shape"], w["transform"], hourly, [1, 2], w["ref_xy"],
                           stamps, w["cfg"], **kw)


def _same(a, b):
    assert np.array_equal(np.isnan(a), np.isnan(b)), "different cells are empty"
    assert np.nanmax(np.abs(a - b)) <= 1e-3


# ------------------------------------------------------------------------------ equivalence
def test_reused_embeddings_equal_one_forward_pass_per_hour(world):
    model = _model(world["cfg"])
    reused, tr = _run(world, model, HOURS)
    plain, tr_plain = _run(world, model, HOURS, reuse_embeddings=False)
    assert reused.shape == (4, ROWS, COLS) and tr == tr_plain and np.isfinite(reused).all()
    _same(reused, plain)
    for h, ts in enumerate(HOURS):                                              # and each hour alone, the old way
        _same(reused[h], _run(world, model, ts, reuse_embeddings=False)[0])


def test_hours_with_different_valid_cells_keep_their_own_gaps(world):
    """Station A (nearest to the top rows) did not report at 05:00, a lag hour of 06:00-08:00 but not of 09:00."""
    values = world["values"].copy()
    values[5, 0] = np.nan
    model = _model(world["cfg"])
    reused, _ = _run(world, model, HOURS, values=values)
    plain, _ = _run(world, model, HOURS, values=values, reuse_embeddings=False)
    _same(reused, plain)
    top, bottom = slice(0, 14), slice(15, ROWS)                                 # nearer A / nearer B
    assert np.isnan(reused[:3, top]).all() and np.isfinite(reused[3, top]).all()      # 06-08 empty, 09 predicted
    assert np.isfinite(reused[:, bottom]).all()


def test_a_chunk_without_any_valid_cell_is_skipped(world):
    values = world["values"].copy()
    values[5, 0] = np.nan                                                       # the first chunk (rows 0-12) is all near A
    reused, _ = _run(world, _model(world["cfg"]), HOURS[:2], values=values)     # 06:00 and 07:00 both read hour 5
    assert np.isnan(reused[:, :12]).all() and np.isfinite(reused[:, 15:]).all()


# ------------------------------------------------------------------------------ shapes and the output grid
def test_a_single_timestamp_returns_a_surface_and_a_list_returns_a_stack(world):
    model = _model(world["cfg"])
    assert _run(world, model, HOURS[0])[0].shape == (ROWS, COLS)
    assert _run(world, model, str(HOURS[0]))[0].shape == (ROWS, COLS)
    assert _run(world, model, HOURS[:1])[0].shape == (1, ROWS, COLS)
    assert _run(world, model, HOURS)[0].shape == (4, ROWS, COLS)


def test_stride_and_window_give_the_same_grid_for_every_hour(world):
    world["cfg"]["inference"].update(stride_px=3, window=[0, 9, 0, 6])
    model = _model(world["cfg"])
    stack, tr = _run(world, model, HOURS)
    single, tr_single = _run(world, model, HOURS[2], reuse_embeddings=False)
    assert stack.shape == (4, 3, 2) and tr == tr_single and tr.a == 30.0
    _same(stack[2], single)


# ------------------------------------------------------------------------------ models that cannot reuse
def test_a_temporal_only_model_takes_the_plain_path_and_gives_the_same_result(world):
    model = _model(world["cfg"], spatial=False)
    assert not cell_inference._supports_reuse(model) and cell_inference._supports_reuse(_model(world["cfg"]))
    stack, _ = _run(world, model, HOURS)
    for h, ts in enumerate(HOURS):
        _same(stack[h], _run(world, model, ts)[0])


class _Stub(torch.nn.Module):
    """Only forward(spatial, temporal), like the stubs of the older tests: predicts the last lag."""
    calls = 0

    def forward(self, spatial, temporal):
        _Stub.calls += 1
        return temporal[:, -1, :1]


def test_a_forward_only_model_takes_the_plain_path(world):
    stub = _Stub()
    assert not cell_inference._supports_reuse(stub)
    stack, _ = _run(world, stub, HOURS)
    for h, ts in enumerate(HOURS):
        _same(stack[h], _run(world, stub, ts)[0])


# ------------------------------------------------------------------------------ fail fast, watchdog
@pytest.mark.parametrize("bad", [START + pd.Timedelta(hours=1), START + pd.Timedelta(hours=40)],
                         ids=["not enough history", "after the last reading"])
def test_an_impossible_hour_stops_the_run_before_any_chunk_work(world, bad, monkeypatch):
    chunks = []
    real = cell_inference.nearest_station_weights
    monkeypatch.setattr(cell_inference, "nearest_station_weights", lambda *a: chunks.append(1) or real(*a))
    _Stub.calls = 0
    with pytest.raises(ValueError, match="not enough history|after the hourly readings"):
        _run(world, _Stub(), [HOURS[0], bad])
    assert _Stub.calls == 0 and chunks == []                                    # not even the first chunk started


def test_the_last_hour_of_the_data_is_still_allowed(world):
    last = START + pd.Timedelta(hours=len(world["values"]))                     # its lags are the final three rows
    assert np.isfinite(_run(world, _Stub(), last)[0]).all()


def test_the_watchdog_hears_from_every_chunk(world, monkeypatch):
    beats = []

    class Guard:
        def beat(self, where=""):
            beats.append(where)

    @contextmanager
    def fake_guarded(label, quiet=False):
        yield Guard()

    monkeypatch.setattr(cell_inference, "guarded", fake_guarded)
    _run(world, _model(world["cfg"]), HOURS)
    assert len(beats) == 3 and beats[0].startswith("chunk 1/3") and "across 4 snapshots" in beats[-1]
