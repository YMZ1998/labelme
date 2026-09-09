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


def test_smooth_polygon_reduces_roughness_without_adding_points() -> None:
    points = np.array(
        [[0, 0], [4, -1], [8, 0], [9, 5], [8, 10], [4, 8], [0, 10], [-1, 5]],
        dtype=np.float64,
    )
    original = points.copy()

    smoothed = smooth_polygon(points)

    assert smoothed.shape == points.shape
    assert _roughness(smoothed) < _roughness(points)
    np.testing.assert_array_equal(points, original)


def test_smooth_polygon_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least three"):
        smooth_polygon([[0, 0], [1, 1]])
