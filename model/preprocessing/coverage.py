"""Does the Master GeoTIFF cover the whole study area (the NCR)?

A sensor outside the raster is skipped by the loader, which silently crops the study area. This module makes that
visible: which sensors are outside (and how far), and, if data.paths.boundary is set, how much of the NCR boundary the
raster covers and how many km are missing on each side.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SIDES = ("west", "east", "south", "north")


def all_sensor_locations(cfg: dict) -> pd.DataFrame:
    """Every sensor of the locations table (inside the raster or not): location_id, name, x, y (projected), lat, lon."""
    import geopandas as gpd

    d = cfg["data"]
    c = d["columns"]
    if d["paths"]["sensors_gpkg"]:
        gdf = gpd.read_file(d["paths"]["sensors_gpkg"])
        ll = gdf.to_crs("EPSG:4326")
        lat, lon = ll.geometry.y.to_numpy(), ll.geometry.x.to_numpy()
        gdf = gdf.to_crs(d["crs"])
    else:
        csvs = d["paths"]["sensor_locations_csv"]
        raw = pd.concat([pd.read_csv(p) for p in ([csvs] if isinstance(csvs, str) else csvs)], ignore_index=True)
        gdf = gpd.GeoDataFrame(raw.copy(), geometry=gpd.points_from_xy(raw[c["lon"]], raw[c["lat"]]),
                               crs="EPSG:4326").to_crs(d["crs"])
        lat, lon = raw[c["lat"]].to_numpy(), raw[c["lon"]].to_numpy()
    names = gdf["name"].astype(str).to_numpy() if "name" in gdf else gdf[c["group"]].astype(str).to_numpy()
    out = pd.DataFrame({c["group"]: gdf[c["group"]].astype("int64").to_numpy(), "name": names,
                        c["x"]: gdf.geometry.x.to_numpy(), c["y"]: gdf.geometry.y.to_numpy(), "lat": lat, "lon": lon})
    return out.drop_duplicates(c["group"]).reset_index(drop=True)


def _beyond(x, y, b) -> dict:
    """Metres a point lies beyond each edge of the raster bounds (0 where it does not)."""
    return {"west": max(0.0, b.left - x), "east": max(0.0, x - b.right),
            "south": max(0.0, b.bottom - y), "north": max(0.0, y - b.top)}


def coverage_report(cfg: dict) -> dict:
    import rasterio
    from pyproj import Transformer

    d = cfg["data"]
    c = d["columns"]
    with rasterio.open(d["paths"]["master_raster"]) as src:
        b, res, crs = src.bounds, float(src.res[0]), src.crs
    to_lonlat = Transformer.from_crs(crs, 4326, always_xy=True)
    (w, s), (e, n) = to_lonlat.transform(b.left, b.bottom), to_lonlat.transform(b.right, b.top)

    sensors = all_sensor_locations(cfg)
    outside = []
    for row in sensors.itertuples(index=False):
        gap = _beyond(getattr(row, c["x"]), getattr(row, c["y"]), b)
        if max(gap.values()) > 0:
            sides = [k for k in SIDES if gap[k] > 0]
            outside.append({"location_id": int(getattr(row, c["group"])), "name": row.name, "lat": float(row.lat),
                            "lon": float(row.lon), "outside_by_m": float(max(gap.values())), "sides": sides})
    report = {
        "raster": {"path": d["paths"]["master_raster"], "crs": crs.to_string(), "resolution_m": res,
                   "bounds": [b.left, b.bottom, b.right, b.top], "lonlat_bounds": [w, s, e, n],
                   "size_km": [(b.right - b.left) / 1000, (b.top - b.bottom) / 1000]},
        "sensors": {"total": int(len(sensors)), "inside": int(len(sensors) - len(outside)), "outside": outside},
        "boundary": None,
    }
    complete = not outside

    if d["paths"].get("boundary") and not Path(d["paths"]["boundary"]).is_file():
        report["boundary_missing"] = d["paths"]["boundary"]           # e.g. a Kaggle clone without data/Boundaries: not an error
    elif d["paths"].get("boundary"):
        import geopandas as gpd
        from shapely.geometry import box

        boundary = gpd.read_file(d["paths"]["boundary"]).to_crs(d["crs"])
        shape = boundary.geometry.union_all() if hasattr(boundary.geometry, "union_all") else boundary.geometry.unary_union
        covered = shape.intersection(box(b.left, b.bottom, b.right, b.top)).area / shape.area
        minx, miny, maxx, maxy = shape.bounds
        missing = {"west": max(0.0, b.left - minx), "east": max(0.0, maxx - b.right),
                   "south": max(0.0, b.bottom - miny), "north": max(0.0, maxy - b.top)}
        report["boundary"] = {"path": d["paths"]["boundary"], "area_km2": shape.area / 1e6,
                              "covered_fraction": float(covered),
                              "missing_km": {k: v / 1000 for k, v in missing.items() if v > 0}}
        complete = complete and covered >= 0.999
    report["complete"] = bool(complete)
    return report


def format_coverage(r: dict) -> str:
    rs, se = r["raster"], r["sensors"]
    w, s, e, n = rs["lonlat_bounds"]
    lines = [f"Master GeoTIFF: {rs['size_km'][0]:.1f} x {rs['size_km'][1]:.1f} km at {rs['resolution_m']:g} m, {rs['crs']}, "
             f"lon {w:.3f}..{e:.3f} E, lat {s:.3f}..{n:.3f} N",
             f"Sensors inside the raster: {se['inside']} of {se['total']}"]
    for o in se["outside"]:
        lines.append(f"  OUTSIDE (skipped): {o['name']} ({o['location_id']}) at {o['lat']:.4f}, {o['lon']:.4f}: "
                     f"{o['outside_by_m'] / 1000:.1f} km beyond the {'/'.join(o['sides'])} edge")
    if r["boundary"]:
        b = r["boundary"]
        lines.append(f"NCR boundary ({b['area_km2']:.0f} km2): {b['covered_fraction']:.1%} covered by the raster")
        for side, km in b["missing_km"].items():
            lines.append(f"  boundary extends {km:.1f} km beyond the {side} edge of the raster")
    elif r.get("boundary_missing"):
        lines.append(f"NCR boundary: {r['boundary_missing']} not found, so coverage of the NCR itself is judged by the sensors only "
                     "(python -m preprocessing.boundaries builds it; the maps are then not clipped to the NCR)")
    else:
        lines.append("NCR boundary: not set (data.paths.boundary), so coverage of the NCR itself is judged by the sensors only")
    lines.append("COVERAGE: " + ("complete" if r["complete"] else "INCOMPLETE: part of the study area is cropped away"))
    return "\n".join(lines)


def check_coverage(cfg: dict, verbose: bool = True) -> dict:
    """Print the coverage report; raise if data.coverage.strict and it is incomplete."""
    report = coverage_report(cfg)
    text = format_coverage(report)
    if verbose:
        print(text)
    if cfg["data"].get("coverage", {}).get("strict") and not report["complete"]:
        raise ValueError("data.coverage.strict: the Master GeoTIFF does not cover the whole study area\n" + text)
    return report
