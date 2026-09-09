from collections.abc import Callable

from PySide6 import QtCore
from PySide6 import QtWidgets


class OnnxSettingsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        keep_classes: list[str],
        minimum_polygon_area: int,
        on_classes_change: Callable[[list[str]], bool],
        on_minimum_area_change: Callable[[int], bool],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("ONNX Settings"))
        self.setModal(False)
        self._on_classes_change = on_classes_change
        self._on_minimum_area_change = on_minimum_area_change

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(self.tr("Classes to keep")))
        class_row = QtWidgets.QHBoxLayout()
        self._checks: dict[str, QtWidgets.QCheckBox] = {}
        for label in ("1", "2", "3"):
            check = QtWidgets.QCheckBox(label)
            check.setChecked(label in keep_classes)
            check.toggled.connect(self._apply)
            class_row.addWidget(check)
            self._checks[label] = check
        class_row.addStretch(1)
        layout.addLayout(class_row)

        area_row = QtWidgets.QHBoxLayout()
        area_row.addWidget(QtWidgets.QLabel(self.tr("Minimum polygon area")))
        self.minimum_area_spinbox = QtWidgets.QSpinBox()
        self.minimum_area_spinbox.setRange(0, 1_000_000)
        self.minimum_area_spinbox.setSuffix(self.tr(" px²"))
        self.minimum_area_spinbox.setSpecialValueText(self.tr("No filtering"))
        self.minimum_area_spinbox.setValue(minimum_polygon_area)
        self.minimum_area_spinbox.valueChanged.connect(self._apply_minimum_area)
        area_row.addWidget(self.minimum_area_spinbox)
        layout.addLayout(area_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    @property
    def keep_classes(self) -> list[str]:
        return [label for label, check in self._checks.items() if check.isChecked()]

    def set_keep_classes(self, values: list[str]) -> None:
        for label, check in self._checks.items():
            with QtCore.QSignalBlocker(check):
                check.setChecked(label in values)

    def set_minimum_polygon_area(self, value: int) -> None:
        with QtCore.QSignalBlocker(self.minimum_area_spinbox):
            self.minimum_area_spinbox.setValue(value)

    def _apply(self) -> None:
        self._on_classes_change(self.keep_classes)

    def _apply_minimum_area(self, value: int) -> None:
        self._on_minimum_area_change(value)
