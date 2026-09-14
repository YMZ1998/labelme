import numpy as np
from PySide6 import QtCore
from PySide6 import QtGui
from pytestqt.qtbot import QtBot

from labelme._widgets.canvas import Canvas
from labelme._widgets.canvas import _DraftShape


def _key_event(
    event_type: QtCore.QEvent.Type,
    key: QtCore.Qt.Key,
    modifiers: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier,
) -> QtGui.QKeyEvent:
    return QtGui.QKeyEvent(event_type, key, modifiers)


def test_space_persistently_switches_ring_side(qtbot: QtBot) -> None:
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_editing(value=False, create_mode="annular_sector")
    points = (
        QtCore.QPointF(10, 10),
        QtCore.QPointF(20, 20),
        QtCore.QPointF(30, 20),
    )
    canvas._current = _DraftShape(
        shape_type="polygon", points=points, point_labels=(1, 1, 1)
    )
    canvas._line = _DraftShape(
        shape_type="polygon",
        points=(points[-1], QtCore.QPointF(40, 10)),
        point_labels=(1, 1),
    )

    canvas.keyPressEvent(
        _key_event(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Space)
    )

    assert canvas._ring_prefer_major_arc is False
    assert canvas._ring_major_arc is False
    assert "small side" in canvas._get_create_mode_message()

    canvas._update_drawing_line(
        pos=QtCore.QPointF(42, 10), is_shift_pressed=True
    )
    assert canvas._ring_major_arc is True

    canvas.keyReleaseEvent(
        _key_event(
            QtCore.QEvent.Type.KeyRelease,
            QtCore.Qt.Key.Key_Shift,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
    )
    assert canvas._ring_major_arc is False


def test_space_generates_reverse_ring_preview_with_boundary_gaps(
    qtbot: QtBot,
) -> None:
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_editing(value=False, create_mode="annular_sector")
    yy, xx = np.indices((241, 241))
    radius = np.hypot(xx - 120, yy - 120)
    mask = (radius >= 35) & (radius <= 95)
    mask[82:88, 190:198] = False
    mask[118:124, 211:218] = False
    mask[152:158, 190:198] = False
    canvas.set_ring_mask(mask, point_spacing=12)
    points = (
        QtCore.QPointF(120, 215),
        QtCore.QPointF(120, 155),
        QtCore.QPointF(120, 85),
    )
    canvas._current = _DraftShape(
        shape_type="polygon", points=points, point_labels=(1, 1, 1)
    )
    canvas._line = _DraftShape(
        shape_type="polygon",
        points=(points[-1], QtCore.QPointF(120, 25)),
        point_labels=(1, 1),
    )

    default_preview = canvas._build_preview_shapes()
    canvas.keyPressEvent(
        _key_event(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Space)
    )
    reverse_preview = canvas._build_preview_shapes()

    assert len(default_preview) == 1
    assert len(reverse_preview) == 1
    assert (
        np.mean(default_preview[0].points[:, 0]) - 120
    ) * (np.mean(reverse_preview[0].points[:, 0]) - 120) < 0
