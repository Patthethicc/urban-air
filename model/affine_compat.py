"""Compose affine transforms on both affine 2.x (`a * b`) and affine 3.x (`a @ b`; `*` is deprecated there)."""
from affine import Affine


def compose(*transforms: Affine) -> Affine:
    """transforms[0] applied last ... transforms[-1] applied first, exactly like `a @ b @ c` / `a * b * c`."""
    result = transforms[0]
    for t in transforms[1:]:
        result = result @ t if hasattr(type(result), "__matmul__") else result * t
    return result
