"""Burn vector metrics into rasters and stack the multi-band Master GeoTIFF (CNN input)."""
import numpy as np


def rasterize_building_density(buildings_gdf, ref_profile, supersample: int = 4) -> np.ndarray:
    """Pixel = proportion of cell area covered by footprints (mean of a `supersample`x finer binary mask)."""
    from affine import Affine

    from affine_compat import compose
    from rasterio.features import rasterize

    h, w, ss = ref_profile["height"], ref_profile["width"], int(supersample)
    fine = rasterize(((geom, 1) for geom in buildings_gdf.geometry), out_shape=(h * ss, w * ss),
                     transform=compose(ref_profile["transform"], Affine.scale(1.0 / ss)), fill=0, dtype="uint8")
    return fine.reshape(h, ss, w, ss).mean(axis=(1, 3), dtype=np.float32)


def rasterize_street_width(streets_gdf, ref_profile, width_col: str = "calculated_width") -> np.ndarray:
    """Buffer centerlines by calculated_width, rasterize; pixel = max width of intersecting segments.
    Shapes are burned widest last so the widest wins where roads overlap."""
    from rasterio.features import rasterize

    g = streets_gdf[streets_gdf[width_col].notna()].copy()
    g["geometry"] = g.geometry.buffer(g[width_col] / 2.0)
    g = g.sort_values(width_col)
    shapes = ((geom, float(width)) for geom, width in zip(g.geometry, g[width_col]))
    return rasterize(shapes, out_shape=(ref_profile["height"], ref_profile["width"]),
                     transform=ref_profile["transform"], fill=0.0, all_touched=True, dtype="float32")


def compute_pixel_aspect_ratio(height: np.ndarray, street_width: np.ndarray, cfg: dict) -> np.ndarray:
    """Mean building height in a 50 m neighbourhood of each road pixel / street width at that pixel."""
    from scipy.ndimage import uniform_filter

    r = cfg["data"]["raster"]
    px = max(1, round(r["aspect_neighborhood_m"] / r["resolution_m"]))
    local_mean = uniform_filter(height.astype("float32"), size=px * 2 + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(street_width > 0, local_mean / street_width, 0.0)
    return ratio.astype("float32")


def compute_height_std(height: np.ndarray, cfg: dict) -> np.ndarray:
    """sigma_H: local standard deviation of building height in a window of
    data.raster.height_std_neighborhood_m around every pixel."""
    from scipy.ndimage import uniform_filter

    r = cfg["data"]["raster"]
    px = max(1, round(r["height_std_neighborhood_m"] / r["resolution_m"]))
    h = height.astype("float64")
    size = px * 2 + 1
    var = uniform_filter(h * h, size=size) - uniform_filter(h, size=size) ** 2
    return np.sqrt(np.maximum(var, 0.0)).astype("float32")


def build_master_geotiff(bands: dict[str, np.ndarray], ref_profile, out_path: str, band_names: list[str]) -> str:
    """Stack height, ndvi, building_density, street_width, aspect_ratio (order from data.yaml raster.band_order)."""
    import rasterio

    profile = ref_profile.copy()
    profile.update(count=len(band_names), dtype="float32", nodata=None, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, name in enumerate(band_names, start=1):
            dst.write(np.nan_to_num(bands[name]).astype("float32"), i)
        dst.descriptions = tuple(band_names)
    return out_path
