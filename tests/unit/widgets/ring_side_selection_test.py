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
