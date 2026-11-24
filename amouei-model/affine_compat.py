"""Compose affine transforms on both affine 2.x (`a * b`) and affine 3.x (`a @ b`; `*` is deprecated there)."""
from affine import Affine


def cell_centers(transform: Affine, rows, cols):
    """Projected (x, y) of the CENTRE of pixels (rows, cols), vectorised: the same numbers as
    rasterio.transform.xy(..., offset="center") but fast and light on memory for 10^7 cells."""
    import numpy as np

    c, r = np.asarray(cols, dtype=np.float64) + 0.5, np.asarray(rows, dtype=np.float64) + 0.5
    return transform.a * c + transform.b * r + transform.c, transform.d * c + transform.e * r + transform.f


def compose(*transforms: Affine) -> Affine:
    """transforms[0] applied last ... transforms[-1] applied first, exactly like `a @ b @ c` / `a * b * c`."""
    result = transforms[0]
    for t in transforms[1:]:
        result = result @ t if hasattr(type(result), "__matmul__") else result * t
    return result
