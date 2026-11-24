"""NCR city boundaries for the maps: GADM 4.1 level-2 polygons (cities and municipalities) of Metropolitan Manila.

    python -m preprocessing.boundaries --src ../data/Boundaries/raw/gadm41_PHL_2.json --out ../data/Boundaries/ncr_cities.gpkg

The source is https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_PHL_2.json.zip (free for academic, non-commercial use;
do not redistribute, so data/ stays out of git). The output has one polygon per city in data.crs with a `city` column that the
maps use for outlines, labels and the clip to the NCR (output/cartography.py).
"""
import argparse
import unicodedata
from pathlib import Path

NCR_NAME_1 = "metropolitanmanila"
CITIES = 17                                    # 16 cities and 1 municipality (Pateros) make up the NCR
DISPLAY = {"kalookancity": "Caloocan", "laspinas": "Las Piñas", "makaticity": "Makati", "malabon": "Malabon",
           "mandaluyong": "Mandaluyong", "manila": "Manila", "marikina": "Marikina", "muntinlupa": "Muntinlupa",
           "navotas": "Navotas", "paranaque": "Parañaque", "pasaycity": "Pasay", "pasigcity": "Pasig", "pateros": "Pateros",
           "quezoncity": "Quezon City", "sanjuan": "San Juan", "taguig": "Taguig", "valenzuela": "Valenzuela"}


def _key(name: str) -> str:
    """'Las Piñas', 'LasPiñas' and 'LAS PIÑAS' all become 'laspinas' (letters only, accents dropped, lower case)."""
    folded = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in folded.lower() if ch.isalpha())


def extract_ncr(src: str, crs: str):
    """The 17 NCR polygons of a GADM level-2 file, in `crs`, with columns city and geometry. Stops unless exactly 17 are found
    and every one has a display name (a different GADM version or a wrong file must not give a quietly wrong outline)."""
    import geopandas as gpd

    gdf = gpd.read_file(src)
    ncr = gdf[gdf["NAME_1"].map(_key) == NCR_NAME_1].copy()
    if len(ncr) != CITIES:
        raise ValueError(f"{src}: expected {CITIES} Metropolitan Manila polygons (NAME_1), found {len(ncr)}")
    ncr["city"] = [DISPLAY.get(k) for k in ncr["NAME_2"].map(_key)]
    if ncr["city"].isna().any():
        raise ValueError(f"unknown NCR city names in {src}: {sorted(ncr.loc[ncr['city'].isna(), 'NAME_2'])}")
    ncr = ncr[["city", "geometry"]].to_crs(crs).sort_values("city").reset_index(drop=True)
    ncr["area_km2"] = ncr.geometry.area / 1e6
    return ncr


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True, help="gadm41_PHL_2.json (unzipped)")
    p.add_argument("--out", required=True, help="output .gpkg")
    p.add_argument("--crs", default="EPSG:32651")
    a = p.parse_args(argv)
    ncr = extract_ncr(a.src, a.crs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ncr.to_file(a.out, driver="GPKG")
    print(f"{len(ncr)} NCR cities, {ncr['area_km2'].sum():.0f} km2, {a.crs} -> {a.out}")


if __name__ == "__main__":
    main()
