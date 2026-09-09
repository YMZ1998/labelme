import numpy as np
import pytest

from labelme._polygon_smoothing import smooth_polygon


def _roughness(points: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.roll(points, 1, axis=0) - 2 * points + np.roll(points, -1, axis=0),
            axis=1,
        ).sum()
    )


def _area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def test_smooth_polygon_reduces_roughness_without_adding_points() -> None:
    points = np.array(
        [[0, 0], [4, -1], [8, 0], [9, 5], [8, 10], [4, 8], [0, 10], [-1, 5]],
        dtype=np.float64,
    )
    original = points.copy()

    smoothed = smooth_polygon(points)

    assert smoothed.shape == points.shape
    assert _roughness(smoothed) < _roughness(points)
    assert abs(_area(smoothed) - _area(points)) / _area(points) < 0.05
    np.testing.assert_array_equal(points, original)


def test_smooth_polygon_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least three"):
        smooth_polygon([[0, 0], [1, 1]])


def test_smooth_polygon_preserves_sharp_corners() -> None:
    points = np.array(
        [[0, 0], [5, -1], [10, 0], [10, 5], [10, 10], [5, 10], [0, 10], [0, 5]],
        dtype=np.float64,
    )

    smoothed = smooth_polygon(points)

    np.testing.assert_array_equal(smoothed[[0, 2, 4, 6]], points[[0, 2, 4, 6]])
    assert smoothed[1, 1] > points[1, 1]


def test_smooth_polygon_limits_vertex_displacement() -> None:
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    radii = 20 + np.where(np.arange(len(angles)) % 2, 2, -2)
    points = np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii))
    median_edge = np.median(
        np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
    )

    smoothed = smooth_polygon(points)

    displacement = np.linalg.norm(smoothed - points, axis=1)
    assert displacement.max() <= median_edge * 0.35 + 1e-9
