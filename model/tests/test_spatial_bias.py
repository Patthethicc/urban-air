import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from reporting.spatial_bias import (aggregate_residuals_by_cell, band_profile, global_moran, knn_weights,
                                    local_moran)


@pytest.fixture
def raster(tmp_path):
    path = tmp_path / "r.tif"
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=2, dtype="float32",
                       crs="EPSG:32651", transform=from_origin(0, 1000, 100, 100)) as dst:
        dst.write(np.ones((2, 10, 10), "float32"), 1) if False else None
        dst.write(np.full((10, 10), 1.0, "float32"), 1)
        dst.write(np.full((10, 10), 2.0, "float32"), 2)
    return str(path)


def _grid_cells(n_side=6):
    """A 6 x 6 lattice of cell centres; residual is +5 on the left half and -5 on the right half."""
    xs, ys = np.meshgrid(np.arange(n_side) * 100.0 + 50, np.arange(n_side) * 100.0 + 50)
    cx, cy = xs.ravel(), ys.ravel()
    resid = np.where(cx < n_side * 50, 5.0, -5.0) + np.random.default_rng(0).normal(0, 0.3, len(cx))
    return pd.DataFrame({"cx": cx, "cy": cy, "mean_residual": resid})


def test_residuals_are_actual_minus_predicted_and_averaged_per_cell(raster):
    oof = pd.DataFrame({"location_id": [1, 1, 2, 3], "y_true": [10.0, 20.0, 30.0, 8.0], "y_pred": [8.0, 14.0, 20.0, 9.0]})
    st = pd.DataFrame({"location_id": [1, 2, 3], "x": [50.0, 60.0, 750.0], "y": [950.0, 940.0, 150.0]})
    cells = aggregate_residuals_by_cell(oof, st, raster, cell_size_m=100.0).set_index("n_stations")
    # stations 1 and 2 share the 100 m cell at the raster's top-left corner: residuals 2, 6, 10 -> mean 6
    shared = cells.loc[2]
    assert shared["mean_residual"] == pytest.approx(6.0) and shared["n_rows"] == 3 and shared["stations"] == "1;2"
    assert cells.loc[1]["mean_residual"] == pytest.approx(-1.0)            # station 3 over-predicted


def test_clustered_residuals_give_a_significant_positive_global_moran():
    cells = _grid_cells()
    w = knn_weights(cells[["cx", "cy"]].to_numpy(), k=4)
    g = global_moran(cells["mean_residual"], w, permutations=499, seed=1)
    assert g["I"] > 0.5 and g["significant"] and g["p_sim"] < 0.01


def test_random_residuals_are_not_significant():
    cells = _grid_cells()
    cells["mean_residual"] = np.random.default_rng(7).normal(0, 1, len(cells))
    w = knn_weights(cells[["cx", "cy"]].to_numpy(), k=4)
    assert not global_moran(cells["mean_residual"], w, permutations=499, seed=1)["significant"]


def test_lisa_labels_under_and_over_prediction_clusters():
    """Positive residual (actual > predicted) = under-prediction = High-High; negative = Low-Low."""
    cells = _grid_cells()
    w = knn_weights(cells[["cx", "cy"]].to_numpy(), k=4)
    lisa = local_moran(cells["mean_residual"], w, permutations=499, seed=1)
    left = cells["cx"] < 300
    assert (lisa.loc[left & (lisa["cluster"] != "Not significant"), "cluster"] == "High-High").all()
    assert (lisa.loc[~left & (lisa["cluster"] != "Not significant"), "cluster"] == "Low-Low").all()
    assert (lisa["cluster"] == "High-High").sum() >= 3 and (lisa["cluster"] == "Low-Low").sum() >= 3


def test_an_esda_without_the_alternative_option_is_refused_with_the_fix(monkeypatch):
    """Kaggle's preinstalled esda can be older than the one this was tested with: say so, instead of a bare TypeError."""
    import esda.moran

    class OldMoranLocal:                                       # the signature of an esda before the `alternative` option
        def __init__(self, y, w, transformation="r", permutations=999, geoda_quads=False, n_jobs=1, keep_simulations=True,
                     seed=None, island_weight=0):
            raise AssertionError("must not be called: the check comes first")

    monkeypatch.setattr(esda.moran, "Moran_Local", OldMoranLocal)
    cells = _grid_cells()
    with pytest.raises(RuntimeError, match=r"esda .* too old.*pip install -U esda"):
        local_moran(cells["mean_residual"], knn_weights(cells[["cx", "cy"]].to_numpy(), k=4), permutations=99)


def test_band_profile_reports_how_clusters_differ_from_all_cells():
    cells = pd.DataFrame({"cluster": ["High-High"] * 3 + ["Low-Low"] * 3})
    means = np.array([[10.0, 1.0]] * 3 + [[2.0, 1.0]] * 3)           # band 0 differs, band 1 constant
    prof = {r["cluster"]: r for r in band_profile(cells, means, ["height", "ndvi"])}
    assert prof["High-High"]["band_mean"]["height"] == 10.0
    assert prof["High-High"]["std_units_from_all_cells"]["height"] > 0 > prof["Low-Low"]["std_units_from_all_cells"]["height"]
    assert prof["High-High"]["std_units_from_all_cells"]["ndvi"] == 0.0
