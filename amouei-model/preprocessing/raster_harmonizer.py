"""Make height (4 m) and NDVI (10 m) share one 10 m x 10 m grid, CRS and NoData extent."""
import numpy as np


def _resampling(name: str):
    from rasterio.enums import Resampling
    return getattr(Resampling, name)


def reproject_raster(src_path: str, dst_path: str, dst_crs: str, resampling: str = "bilinear",
                     resolution: float | None = None) -> str:
    """Reproject to dst_crs. Pass `resolution` (metres) to get exact square pixels of that size, so
    height (4 m) and NDVI (10 m) land on the same 10 m grid in one bilinear step."""
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds,
                                                              resolution=resolution)
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(source=rasterio.band(src, band), destination=rasterio.band(dst, band),
                          src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                          dst_crs=dst_crs, resampling=_resampling(resampling))
    return dst_path


def resample_to_resolution(src_path: str, dst_path: str, target_resolution_m: float = 10.0,
                           resampling: str = "bilinear") -> str:
    """4 m height -> 10 m. Bilinear (not nearest) to keep continuous height transitions.
    The raster must already be in a metric CRS."""
    import rasterio

    with rasterio.open(src_path) as src:
        new_w = max(1, round(src.width * src.res[0] / target_resolution_m))
        new_h = max(1, round(src.height * src.res[1] / target_resolution_m))
        data = src.read(out_shape=(src.count, new_h, new_w), resampling=_resampling(resampling))
        transform = src.transform * src.transform.scale(src.width / new_w, src.height / new_h)
        profile = src.profile.copy()
        profile.update(transform=transform, width=new_w, height=new_h)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)
    return dst_path


def clip_and_mask(src_path: str, boundary_gdf, dst_path: str, nodata: float = -9999.0) -> str:
    """Clip to the NCR boundary; pixels outside it become NoData."""
    import rasterio
    from rasterio.mask import mask as rio_mask

    with rasterio.open(src_path) as src:
        nd = src.nodata if src.nodata is not None else nodata
        boundary = boundary_gdf.to_crs(src.crs)
        image, transform = rio_mask(src, boundary.geometry, crop=True, nodata=nd)
        profile = src.profile.copy()
        profile.update(height=image.shape[1], width=image.shape[2], transform=transform, nodata=nd)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(image)
    return dst_path


def align_to_reference(src_path: str, ref_path: str, dst_path: str, resampling: str = "bilinear") -> str:
    """Warp `src` onto the exact grid of `ref` (CRS, origin, size) and copy ref's NoData mask, so the
    layers can be stacked pixel for pixel."""
    import rasterio
    from rasterio.warp import reproject

    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        profile = ref.profile.copy()
        profile.update(count=src.count, dtype="float32", nodata=nodata)
        ref_invalid = ref.read(1, masked=True).mask
        with rasterio.open(dst_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                out = np.full((ref.height, ref.width), nodata, dtype=np.float32)
                reproject(source=rasterio.band(src, b), destination=out, src_transform=src.transform,
                          src_crs=src.crs, src_nodata=src.nodata, dst_transform=ref.transform,
                          dst_crs=ref.crs, dst_nodata=nodata, resampling=_resampling(resampling))
                out[ref_invalid] = nodata
                dst.write(out, b)
    return dst_path
