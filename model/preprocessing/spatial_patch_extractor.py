"""Cut [C, H, W] patches from the morphology raster, centred on each station."""
import numpy as np


def station_to_pixel(lat: float, lon: float, transform) -> tuple[int, int]:
    """Map station coordinates to (row, col) in the raster."""
    # TODO
    raise NotImplementedError


def extract_patches(raster_path: str, stations, cfg: dict) -> dict[str, np.ndarray]:
    """Return {station_id: patch[C, H, W]}; handle stations near the raster edge (pad)."""
    # TODO
    raise NotImplementedError
