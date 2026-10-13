"""Raw vector ingestion and sanitation (buildings, OSM streets) -> metric CRS."""
import geopandas as gpd
import numpy as np
import pandas as pd


def repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix self-intersections / open rings (zero-distance buffer, mirrors the QGIS fix); drop empties."""
    gdf = gdf.copy()
    bad = ~gdf.geometry.is_valid
    if bad.any():
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)
    return gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]


def _first_number(series: pd.Series) -> pd.Series:
    """OSM tags are strings such as '6', '6 m' or '2;3': take the first number."""
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


def impute_street_widths(streets: gpd.GeoDataFrame, cfg: dict) -> gpd.GeoDataFrame:
    """Add `calculated_width`: explicit width -> lanes * lane_width_m -> functional-class fallback,
    clipped to [min_width_m, max_width_m]. OSM `highway` may be a list; use the first."""
    r = cfg["data"]["raster"]
    gdf = streets.copy()
    nan = pd.Series(np.nan, index=gdf.index)
    width = _first_number(gdf["width"]) if "width" in gdf else nan
    lanes = _first_number(gdf["lanes"]) if "lanes" in gdf else nan

    calculated = width.fillna(lanes * r["lane_width_m"])
    if "highway" in gdf:
        cls = gdf["highway"].apply(lambda v: v[0] if isinstance(v, (list, tuple)) else v)
        calculated = calculated.fillna(cls.map(r["functional_class_widths"]))
    calculated = calculated.fillna(r["functional_class_widths"]["unclassified"])
    gdf["calculated_width"] = calculated.clip(lower=r["min_width_m"], upper=r["max_width_m"])
    return gdf


def reproject_layer(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """EPSG:4326 -> EPSG:32651. Raise if the layer has no CRS."""
    if gdf.crs is None:
        raise ValueError("layer has no CRS; cannot safely reproject")
    return gdf if str(gdf.crs) == str(target_crs) else gdf.to_crs(target_crs)


def load_buildings_from_wkt_csv(path: str, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Google Open Buildings ships as CSV with a WKT geometry column."""
    from shapely import wkt

    df = pd.read_csv(path)
    col = next((c for c in df.columns if c.lower() in ("geometry", "wkt", "geometry_wkt")), None)
    if col is None:
        raise KeyError(f"no WKT column in {path}; columns: {list(df.columns)}")
    geom = df[col].apply(wkt.loads)
    return gpd.GeoDataFrame(df.drop(columns=[col]), geometry=geom, crs=crs)


def ingest_and_sanitize(cfg: dict):
    """Return (buildings_gdf, streets_gdf), repaired, width-imputed, reprojected to data.crs."""
    p, crs = cfg["data"]["paths"], cfg["data"]["crs"]
    buildings = reproject_layer(repair_geometries(load_buildings_from_wkt_csv(p["buildings_csv"])), crs)
    streets = reproject_layer(impute_street_widths(gpd.read_file(p["streets"]), cfg), crs)
    return buildings, streets
