from pytestqt.qtbot import QtBot

from labelme._widgets._onnx_settings_dialog import OnnxSettingsDialog


def test_class_selection_applies_immediately(qtbot: QtBot) -> None:
    applied: list[list[str]] = []
    dialog = OnnxSettingsDialog(
        keep_classes=["1", "2", "3"],
        minimum_polygon_area=0,
        on_classes_change=lambda values: applied.append(values) is None,
        on_minimum_area_change=lambda _value: True,
    )
    qtbot.addWidget(dialog)

    dialog._checks["2"].setChecked(False)

    assert dialog.keep_classes == ["1", "3"]
    assert applied == [["1", "3"]]


def test_minimum_polygon_area_applies_immediately(qtbot: QtBot) -> None:
    applied: list[int] = []
    dialog = OnnxSettingsDialog(
        keep_classes=["1", "2", "3"],
        minimum_polygon_area=0,
        on_classes_change=lambda _values: True,
        on_minimum_area_change=lambda value: applied.append(value) is None,
    )
    qtbot.addWidget(dialog)

    dialog.minimum_area_spinbox.setValue(250)

    assert applied == [250]
