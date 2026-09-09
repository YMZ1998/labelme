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
def test_shift_select_from_draw_mode_switches_to_edit(
    *,
    qtbot: QtBot,
    main_win: MainWinFactory,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((80, 80), 127, dtype=np.uint8)).save(image_path)
    win = main_win(file_or_dir=image_path, config_overrides={"auto_save": False})
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    shape = Shape(
        label="1",
        shape_type="polygon",
        points=[[10, 10], [60, 10], [60, 60], [10, 60]],
        closed=True,
    )
    win._load_shapes([shape], replace=True)
    win._switch_canvas_mode(edit=False, create_mode="polygon")
    canvas = win._canvas_widgets.canvas
    point = image_to_widget_pos(
        canvas=canvas,
        image_pos=QtCore.QPointF(30, 30),
    )

    qtbot.mouseMove(canvas, point)
    qtbot.mouseClick(
        canvas,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
        pos=point,
    )

    assert canvas.mode.name == "EDIT"
    assert canvas.selected_shapes == [shape]

    vertex = image_to_widget_pos(
        canvas=canvas,
        image_pos=QtCore.QPointF(10, 10),
    )
    moved_vertex = image_to_widget_pos(
        canvas=canvas,
        image_pos=QtCore.QPointF(14, 14),
    )
    qtbot.mouseMove(canvas, vertex)
    qtbot.waitUntil(lambda: canvas._hovered_vertex == 0)
    qtbot.mousePress(canvas, QtCore.Qt.MouseButton.LeftButton, pos=vertex)
    qtbot.mouseMove(canvas, moved_vertex)
    qtbot.mouseRelease(canvas, QtCore.Qt.MouseButton.LeftButton, pos=moved_vertex)

    np.testing.assert_allclose(shape.points[0], [14, 14], atol=1)
    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)
