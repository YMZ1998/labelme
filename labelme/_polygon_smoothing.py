from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

from ._shape import MIN_POLYGON_POINT_COUNT

STRAIGHT_ANGLE_DEGREES: Final = 180.0


def smooth_polygon(
    points: npt.ArrayLike,
    *,
    iterations: int = 2,
    forward_factor: float = 0.5,
    reverse_factor: float = -0.53,
    corner_preservation_angle: float = 120.0,
    full_smoothing_angle: float = 165.0,
    maximum_displacement_ratio: float = 0.35,
) -> npt.NDArray[np.float64]:
    """Smooth a closed polygon while protecting corners and limiting distortion."""
    result = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    if len(result) < MIN_POLYGON_POINT_COUNT:
        raise ValueError("A polygon needs at least three points")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not np.isfinite(result).all():
        raise ValueError("Polygon points must be finite")
    if not (
        0
        < corner_preservation_angle
        < full_smoothing_angle
        < STRAIGHT_ANGLE_DEGREES
    ):
        raise ValueError("Smoothing angles must be ordered between 0 and 180")
    if maximum_displacement_ratio <= 0:
        raise ValueError("Maximum displacement ratio must be positive")

    original = result.copy()
    weights = _smoothing_weights(
        original,
        corner_angle=corner_preservation_angle,
        full_angle=full_smoothing_angle,
    )
    edge_lengths = np.linalg.norm(original - np.roll(original, 1, axis=0), axis=1)
    positive_edges = edge_lengths[edge_lengths > np.finfo(np.float64).eps]
    if len(positive_edges) == 0:
        return original
    maximum_displacement = float(np.median(positive_edges)) * maximum_displacement_ratio

    for _ in range(iterations):
        for factor in (forward_factor, reverse_factor):
            neighbours = (np.roll(result, 1, axis=0) + np.roll(result, -1, axis=0)) / 2
            result = result + weights[:, None] * factor * (neighbours - result)
            result = _limit_displacement(
                points=result,
                original=original,
                maximum=maximum_displacement,
            )

    original_area = abs(_signed_area(original))
    smoothed_area = abs(_signed_area(result))
    if original_area > 0 and smoothed_area > 0:
        center = result.mean(axis=0)
        scale = math.sqrt(original_area / smoothed_area)
        area_corrected = center + (result - center) * scale
        result += weights[:, None] * (area_corrected - result)
        result = _limit_displacement(
            points=result,
            original=original,
            maximum=maximum_displacement,
        )
    return result


def _smoothing_weights(
    points: npt.NDArray[np.float64], *, corner_angle: float, full_angle: float
) -> npt.NDArray[np.float64]:
    previous = np.roll(points, 1, axis=0) - points
    following = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
    cosine = np.divide(
        np.sum(previous * following, axis=1),
        lengths,
        out=np.ones(len(points), dtype=np.float64),
        where=lengths > np.finfo(np.float64).eps,
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.clip((angles - corner_angle) / (full_angle - corner_angle), 0.0, 1.0)


def _signed_area(points: npt.NDArray[np.float64]) -> float:
    x, y = points[:, 0], points[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _limit_displacement(
    *,
    points: npt.NDArray[np.float64],
    original: npt.NDArray[np.float64],
    maximum: float,
) -> npt.NDArray[np.float64]:
    displacement = points - original
    distances = np.linalg.norm(displacement, axis=1)
    over_limit = distances > maximum
    points[over_limit] = original[over_limit] + displacement[over_limit] * (
        maximum / distances[over_limit]
    )[:, None]
    return points
