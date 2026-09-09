import numpy as np
import pytest

from labelme._polygon_simplification import simplify_polygon


def test_simplify_polygon_removes_redundant_edge_points() -> None:
    points = np.array(
        [
            [0, 0],
            [3, 0],
            [7, 0],
            [10, 0],
            [10, 3],
            [10, 7],
            [10, 10],
            [7, 10],
            [3, 10],
            [0, 10],
            [0, 7],
            [0, 3],
        ],
        dtype=np.float64,
    )
    original = points.copy()

    simplified = simplify_polygon(points)

    assert len(simplified) == 4
    np.testing.assert_array_equal(points, original)


def test_simplify_polygon_keeps_minimum_vertex_count() -> None:
    triangle = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float64)

    assert len(simplify_polygon(triangle)) == 3


def test_smaller_tolerance_retains_more_points() -> None:
    points = np.array(
        [[0, 0], [2, 0.4], [4, 0], [6, 0.4], [8, 0], [8, 8], [0, 8]],
        dtype=np.float64,
    )

    detailed = simplify_polygon(points, tolerance=0.1)
    simplified = simplify_polygon(points, tolerance=0.5)

    assert len(detailed) > len(simplified)


def test_simplify_polygon_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least three"):
        simplify_polygon([[0, 0], [1, 1]])
