"""Cut [C, H, W] patches from the Master GeoTIFF around each sensor (raster is the ONLY source of pixels)."""
import numpy as np


def extract_patch(raster, x: float, y: float, extent_m: float, resolution_m: float) -> np.ndarray:
    """Zero-padded (C, H, W) crop centred on projected (x, y) via raster.index(x, y).
    H = W = max(1, round(extent_m / resolution_m)); `resolution_m` is the raster's real resolution.
    NoData / NaN pixels become 0."""
    from rasterio.windows import Window

    px = max(1, int(round(extent_m / resolution_m)))
    row, col = raster.index(x, y)
    r0, c0 = row - px // 2, col - px // 2
    r1, c1 = r0 + px, c0 + px

    rr0, rr1 = max(r0, 0), min(r1, raster.height)
    cc0, cc1 = max(c0, 0), min(c1, raster.width)

    patch = np.zeros((raster.count, px, px), dtype=np.float32)
    if rr1 > rr0 and cc1 > cc0:
        data = raster.read(window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0)).astype(np.float32)
        if raster.nodata is not None:
            data[data == np.float32(raster.nodata)] = 0.0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        patch[:, rr0 - r0:rr0 - r0 + data.shape[1], cc0 - c0:cc0 - c0 + data.shape[2]] = data
    return patch


def add_cell_columns(stations, raster_path: str):
    """Add the raster cell each sensor falls in: cell_row, cell_col and the projected CELL-CENTRE
    coordinates cell_x, cell_y. The CNN patch is centred on this cell, and GeoCP measures distances
    between cell centres (Validation.txt / Section 4.5)."""
    import rasterio
    from rasterio.transform import rowcol, xy

    out = stations.copy()
    with rasterio.open(raster_path) as src:
        rows, cols = rowcol(src.transform, out["x"].to_numpy(), out["y"].to_numpy())
        cx, cy = xy(src.transform, rows, cols, offset="center")
    out["cell_row"], out["cell_col"] = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
    out["cell_x"], out["cell_y"] = np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)
    return out


def extract_patches(raster_path: str, stations, cfg: dict) -> dict:
    """{location_id: patch[C, H, W]}. A sensor's patch is static, so compute once per sensor.
    `stations` has columns location_id, x, y (see spatial_lag.station_coordinates)."""
    import rasterio

    extent = cfg["data"]["patch"]["extent_m"]
    out = {}
    with rasterio.open(raster_path) as src:
        res = float(src.res[0])
        for row in stations.itertuples(index=False):
            out[int(row.location_id)] = extract_patch(src, row.x, row.y, extent, res)
    return out
