from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pytestqt.qtbot import QtBot

from labelme._shape import Shape

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata


@pytest.mark.gui
def test_smooth_selected_polygon_and_undo(
    *,
    qtbot: QtBot,
    main_win: MainWinFactory,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((64, 64), 127, dtype=np.uint8)).save(image_path)
    win = main_win(file_or_dir=image_path, config_overrides={"auto_save": False})
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    original = np.array(
        [[5, 5], [30, 3], [58, 7], [55, 32], [59, 58], [30, 53], [4, 58]],
        dtype=np.float64,
    )
    shape = Shape(label="1", shape_type="polygon", points=original, closed=True)
    win._load_shapes([shape], replace=True)
    win._canvas_widgets.canvas.select_shapes(shapes=[shape])

    assert win._actions.smooth_polygon.isEnabled()
    assert win._actions.smooth_polygon.shortcut().toString() == "Shift+S"
    win._actions.smooth_polygon.trigger()

    assert len(shape.points) == len(original)
    assert not np.array_equal(shape.points, original)
    assert win._is_changed

    win.undo_shape_edit()
    restored = win._canvas_widgets.canvas.shapes[0]
    np.testing.assert_allclose(restored.points, original)
    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_simplify_selected_polygon_and_undo(
    *,
    qtbot: QtBot,
    main_win: MainWinFactory,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((64, 64), 127, dtype=np.uint8)).save(image_path)
    win = main_win(file_or_dir=image_path, config_overrides={"auto_save": False})
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    original = np.array(
        [
            [5, 5],
            [20, 5],
            [40, 5],
            [58, 5],
            [58, 25],
            [58, 58],
            [35, 58],
            [5, 58],
            [5, 30],
        ],
        dtype=np.float64,
    )
    shape = Shape(label="1", shape_type="polygon", points=original, closed=True)
    win._load_shapes([shape], replace=True)
    win._canvas_widgets.canvas.select_shapes(shapes=[shape])

    assert win._actions.simplify_polygon.isEnabled()
    assert win._actions.simplify_polygon.shortcut().toString() == "Shift+D"
    win._actions.simplify_polygon.trigger()

    assert 3 <= len(shape.points) < len(original)
    assert win._is_changed

    win.undo_shape_edit()
    restored = win._canvas_widgets.canvas.shapes[0]
    np.testing.assert_allclose(restored.points, original)
    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_merge_two_selected_polygons_and_undo(
    *,
    qtbot: QtBot,
    main_win: MainWinFactory,
    tmp_path: Path,
    pause: bool,
) -> None:
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((64, 64), 127, dtype=np.uint8)).save(image_path)
    win = main_win(file_or_dir=image_path, config_overrides={"auto_save": False})
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    first = Shape(
        label="1",
        description="kept",
        shape_type="polygon",
        points=[[5, 5], [35, 5], [35, 35], [5, 35]],
        closed=True,
    )
    second = Shape(
        label="2",
        shape_type="polygon",
        points=[[25, 25], [55, 25], [55, 55], [25, 55]],
        closed=True,
    )
    win._load_shapes([first, second], replace=True)
    win._canvas_widgets.canvas.select_shapes(shapes=[first, second])

    assert win._actions.merge_polygons.isEnabled()
    assert win._actions.merge_polygons.shortcut().toString() == "Ctrl+Alt+M"
    win._actions.merge_polygons.trigger()

    assert win._canvas_widgets.canvas.shapes == [first]
    assert first.label == "1"
    assert first.description == "kept"
    assert len(win._docks.label_list) == 1
    assert win._is_changed

    win.undo_shape_edit()
    assert len(win._canvas_widgets.canvas.shapes) == 2
    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)
