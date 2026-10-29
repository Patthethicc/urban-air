"""NCR city boundaries for the maps: GADM 4.1 level-2 polygons of Metropolitan Manila.

    python -m preprocessing.boundaries                     # downloads GADM, writes data/ncr_cities.gpkg
    python -m preprocessing.boundaries --src gadm41_PHL_2.json --out data/ncr_cities.gpkg

The source is https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_PHL_2.json.zip (free for
academic, non-commercial use; do not redistribute it, which is why data/ stays out of git).

The output has one polygon per city in EPSG:32651 with a `city` column, which is what
maps/cnn_maps.py uses for the clip, the outlines and the labels. Same 17 cities and the same
display names as the CNN-LSTM branch (model/preprocessing/boundaries.py), so the two figures
share one boundary.
"""
import argparse
import io
import os
import unicodedata
import urllib.request
import zipfile

NCR_NAME_1 = "metropolitanmanila"
CITIES = 17                                   # 16 cities and 1 municipality (Pateros) make up the NCR
DISPLAY = {"kalookancity": "Caloocan", "laspinas": "Las Piñas", "makaticity": "Makati",
           "malabon": "Malabon", "mandaluyong": "Mandaluyong", "manila": "Manila",
           "marikina": "Marikina", "muntinlupa": "Muntinlupa", "navotas": "Navotas",
           "paranaque": "Parañaque", "pasaycity": "Pasay", "pasigcity": "Pasig",
           "pateros": "Pateros", "quezoncity": "Quezon City", "sanjuan": "San Juan",
           "taguig": "Taguig", "valenzuela": "Valenzuela"}
URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_PHL_2.json.zip"


def _key(name: str) -> str:
    """'Las Piñas', 'LasPiñas' and 'LAS PIÑAS' all become 'laspinas'."""
    folded = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in folded.lower() if ch.isalpha())


def download(dest_dir: str = ".gadm") -> str:
    """Fetch and unzip GADM level 2, returning the path to the .json."""
    os.makedirs(dest_dir, exist_ok=True)
    json_path = os.path.join(dest_dir, "gadm41_PHL_2.json")
    if os.path.exists(json_path):
        return json_path
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dest_dir)
    return json_path


def extract_ncr(src: str, crs: str = "EPSG:32651"):
    """The 17 NCR polygons with columns city and geometry. Stops unless exactly 17 are found and
    every one has a display name: a different GADM version must not give a quietly wrong outline."""
    import geopandas as gpd

    gdf = gpd.read_file(src)
    ncr = gdf[gdf["NAME_1"].map(_key) == NCR_NAME_1].copy()
    if len(ncr) != CITIES:
        raise ValueError("%s: expected %d Metropolitan Manila polygons (NAME_1), found %d"
                         % (src, CITIES, len(ncr)))
    ncr["city"] = [DISPLAY.get(k) for k in ncr["NAME_2"].map(_key)]
    if ncr["city"].isna().any():
        raise ValueError("unknown NCR city names in %s: %s"
                         % (src, sorted(ncr.loc[ncr["city"].isna(), "NAME_2"])))
    ncr = ncr[["city", "geometry"]].to_crs(crs).sort_values("city").reset_index(drop=True)
    ncr["area_km2"] = ncr.geometry.area / 1e6
    return ncr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=None, help="gadm41_PHL_2.json (downloaded when omitted)")
    ap.add_argument("--out", default=os.path.join("data", "ncr_cities.gpkg"))
    ap.add_argument("--crs", default="EPSG:32651")
    a = ap.parse_args(argv)

    src = a.src or download()
    ncr = extract_ncr(src, a.crs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ncr.to_file(a.out, driver="GPKG")
    print("%d NCR cities, %.0f km2, %s -> %s" % (len(ncr), ncr["area_km2"].sum(), a.crs, a.out))


if __name__ == "__main__":
    main()
