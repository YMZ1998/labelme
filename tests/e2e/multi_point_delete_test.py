from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from labelme._shape import Shape

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import image_to_widget_pos
from .conftest import show_window_and_wait_for_imagedata


@pytest.mark.gui
def test_ctrl_click_arc_endpoints_then_remove_and_undo(
    *,
    qtbot: QtBot,
    main_win: MainWinFactory,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((100, 100), 127, dtype=np.uint8)).save(image_path)
    win = main_win(file_or_dir=image_path, config_overrides={"auto_save": False})
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    original = np.array(
        [[15, 15], [50, 10], [85, 15], [90, 50], [85, 85], [50, 90], [15, 85]],
        dtype=np.float64,
    )
    shape = Shape(label="1", shape_type="polygon", points=original, closed=True)
    win._load_shapes([shape], replace=True)
    canvas = win._canvas_widgets.canvas
    canvas.select_shapes(shapes=[shape])

    for index in (1, 3):
        point = QtCore.QPointF(*shape.points[index])
        widget_pos = image_to_widget_pos(canvas=canvas, image_pos=point)
        qtbot.mouseMove(canvas, widget_pos)
        qtbot.waitUntil(lambda index=index: canvas._hovered_vertex == index)
        qtbot.mouseClick(
            canvas,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
            pos=widget_pos,
        )

    assert canvas._selected_vertex_indices == {1, 3}
    win._actions.remove_point.trigger()

    np.testing.assert_allclose(shape.points, original[[0, 4, 5, 6]])
    assert canvas._selected_vertex_indices == set()
    assert win._is_changed

    win.undo_shape_edit()
    np.testing.assert_allclose(canvas.shapes[0].points, original)
    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)
