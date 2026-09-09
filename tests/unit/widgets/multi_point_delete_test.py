import numpy as np
from pytestqt.qtbot import QtBot

from labelme._shape import Shape
from labelme._widgets.canvas import Canvas
from labelme._widgets.canvas import _shortest_polygon_vertex_arc


def test_shortest_polygon_vertex_arc_crosses_closing_edge() -> None:
    assert _shortest_polygon_vertex_arc(point_count=8, endpoints=(1, 6)) == [0, 1, 6, 7]


def test_shortest_polygon_vertex_arc_uses_forward_arc_for_tie() -> None:
    assert _shortest_polygon_vertex_arc(point_count=8, endpoints=(1, 5)) == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_two_selected_vertices_delete_shortest_arc(qtbot: QtBot) -> None:
    canvas = Canvas()
    qtbot.addWidget(canvas)
    shape = Shape(
        label="1",
        shape_type="polygon",
        points=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0]],
        closed=True,
    )
    canvas.load_shapes(shapes=[shape], replace=True)
    canvas._selected_vertices_shape = shape
    canvas._selected_vertex_indices = {1, 6}

    assert canvas.remove_selected_point() is True
    np.testing.assert_array_equal(
        shape.points,
        np.asarray([[2, 0], [3, 0], [4, 0], [5, 0]]),
    )


def test_multi_point_delete_preserves_polygon_minimum(qtbot: QtBot) -> None:
    canvas = Canvas()
    qtbot.addWidget(canvas)
    shape = Shape(
        label="1",
        shape_type="polygon",
        points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        closed=True,
    )
    canvas.load_shapes(shapes=[shape], replace=True)
    canvas._selected_vertices_shape = shape
    canvas._selected_vertex_indices = {0, 1}
    original = shape.points.copy()

    assert canvas.remove_selected_point() is False
    np.testing.assert_array_equal(shape.points, original)
