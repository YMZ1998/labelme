from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._shape import MIN_POLYGON_POINT_COUNT


def smooth_polygon(
    points: npt.ArrayLike,
    *,
    iterations: int = 2,
    forward_factor: float = 0.5,
    reverse_factor: float = -0.53,
) -> npt.NDArray[np.float64]:
    """Smooth a closed polygon with Taubin passes while preserving point count."""
    result = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    if len(result) < MIN_POLYGON_POINT_COUNT:
        raise ValueError("A polygon needs at least three points")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not np.isfinite(result).all():
        raise ValueError("Polygon points must be finite")

    for _ in range(iterations):
        for factor in (forward_factor, reverse_factor):
            neighbours = (np.roll(result, 1, axis=0) + np.roll(result, -1, axis=0)) / 2
            result = result + factor * (neighbours - result)
    return result
