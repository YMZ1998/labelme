from __future__ import annotations

import numpy as np
import numpy.typing as npt
from skimage.measure import approximate_polygon

from ._shape import MIN_POLYGON_POINT_COUNT


def simplify_polygon(
    points: npt.ArrayLike,
    *,
    tolerance: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Reduce vertices in a closed polygon within a pixel error tolerance."""
    polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < MIN_POLYGON_POINT_COUNT:
        raise ValueError("A polygon needs at least three points")
    if tolerance <= 0:
        raise ValueError("Simplification tolerance must be positive")
    if not np.isfinite(polygon).all():
        raise ValueError("Polygon points must be finite")

    closed = np.vstack([polygon, polygon[0]])
    simplified = approximate_polygon(closed, tolerance=tolerance)
    if np.array_equal(simplified[0], simplified[-1]):
        simplified = simplified[:-1]
    if len(simplified) < MIN_POLYGON_POINT_COUNT:
        return polygon.copy()
    return np.asarray(simplified, dtype=np.float64)
