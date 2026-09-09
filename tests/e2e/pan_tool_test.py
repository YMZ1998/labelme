from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import drag_canvas
from .conftest import show_window_and_wait_for_imagedata


@pytest.mark.gui
def test_pan_tool_uses_left_button_and_exits_for_drawing(
    *,
    qtbot: QtBot,
    main_win: MainWinFactory,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((128, 128), 127, dtype=np.uint8)).save(image_path)
    win = main_win(file_or_dir=image_path)
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    canvas = win._canvas_widgets.canvas
    deltas: list[QtCore.QPoint] = []
    canvas.pan_request.connect(deltas.append)

    win._actions.pan_mode.trigger()
    assert win._actions.pan_mode.isChecked()
    start = QtCore.QPoint(canvas.width() // 2, canvas.height() // 2)
    end = start + QtCore.QPoint(40, 40)
    drag_canvas(
        qtbot=qtbot,
        canvas=canvas,
        button=QtCore.Qt.MouseButton.LeftButton,
        start=start,
        end=end,
    )

    assert sum(point.x() for point in deltas) == 40
    assert sum(point.y() for point in deltas) == 40

    win._actions.create_mode.trigger()
    assert not win._actions.pan_mode.isChecked()

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)
