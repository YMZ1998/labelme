from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._automation._geometry import compute_polygons_from_mask
from ._utils.shape import shape_to_mask


def merge_polygons(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    image_shape: tuple[int, int],
    detail: int = 80,
) -> npt.NDArray[np.float64]:
    """Return the single polygon formed by the raster union of two polygons."""
    mask = shape_to_mask(
        image_shape,
        np.asarray(first, dtype=np.float64).tolist(),
        shape_type="polygon",
    )
    mask |= shape_to_mask(
        image_shape,
        np.asarray(second, dtype=np.float64).tolist(),
        shape_type="polygon",
    )
    polygons = compute_polygons_from_mask(mask=mask, detail=detail)
    if len(polygons) != 1:
        raise ValueError("The two polygons must overlap or touch to form one polygon")
    return np.asarray(polygons[0], dtype=np.float64)
