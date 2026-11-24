"""What each band of the Master GeoTIFF is, in words and units: used by the Grad-CAM figures and the morphology sheet."""

# name -> (unit shown in titles, what it measures, colour map of the morphology sheet)
BAND_INFO = {
    "building_height_mean": ("m", "mean building height in the cell", "Greys"),
    "building_height_stddev": ("m", "how much building heights vary (roughness of the skyline)", "Purples"),
    "ndvi_mean": ("NDVI, -1 to 1", "greenness of the vegetation (higher = more, denser green)", "Greens"),
    "building_density": ("share of area", "share of the ground covered by buildings", "Oranges"),
    "street_aspect_ratio_mean": ("height / width", "depth of the street canyon: building height divided by street width", "Blues"),
}


def unit_of(band: str) -> str:
    return BAND_INFO.get(band, ("", "", ""))[0]


def meaning_of(band: str) -> str:
    return BAND_INFO.get(band, ("", band, ""))[1]


def cmap_of(band: str) -> str:
    return BAND_INFO.get(band, ("", "", "Greys"))[2] or "Greys"


def title_of(band: str) -> str:
    """'building_height_mean' -> 'building height mean\n(m)'"""
    unit = unit_of(band)
    return band.replace("_", " ") + (f"\n({unit})" if unit else "")
