from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

from .. import _utils
from .._ring_config import load_ring_point_spacing
from .._ring_segmentation import estimate_inner_radius
from .._ring_segmentation import fit_imaging_circle
from .._ring_segmentation import mask_contours
from .._ring_segmentation import ring_grayscale
from .._ring_segmentation import trace_imaging_ring

RADIUS_SETTING = "ringContour/radiusPercent"
SMOOTHNESS_SETTING = "ringContour/smoothness"
POINT_SPACING_SETTING = "ringContour/pointSpacing"


@dataclass(frozen=True)
class RingContourParameters:
    radius_percent: int
    smoothness: int
    point_spacing: int


def load_ring_contour_parameters(
    settings: QtCore.QSettings | None, *, default_radius_percent: int
) -> RingContourParameters:
    return RingContourParameters(
        radius_percent=_stored_int(
            settings, RADIUS_SETTING, default_radius_percent, minimum=5, maximum=60
        ),
        smoothness=_stored_int(
            settings, SMOOTHNESS_SETTING, 5, minimum=1, maximum=20
        ),
        point_spacing=_stored_int(
            settings,
            POINT_SPACING_SETTING,
            load_ring_point_spacing(),
            minimum=2,
            maximum=500,
        ),
    )


def _stored_int(
    settings: QtCore.QSettings | None,
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if settings is None:
        return default
    try:
        value = int(settings.value(key, default))
    except (TypeError, ValueError):
        return default
    return value if minimum <= value <= maximum else default


class RingContourDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        image: QtGui.QImage,
        parent: QtWidgets.QWidget | None,
        settings: QtCore.QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle(self.tr("Extract ring contour"))
        self._image = image
        self._gray = ring_grayscale(_utils.img_qt_to_rgb_arr(image))
        try:
            self._circle = fit_imaging_circle(self._gray)
            radius_percent = round(
                100 * estimate_inner_radius(self._gray, self._circle) / self._circle[2]
            )
        except ValueError:
            self._circle = None
            radius_percent = 25
        parameters = load_ring_contour_parameters(
            settings, default_radius_percent=radius_percent
        )
        self.mask = None
        layout = QtWidgets.QVBoxLayout(self)
        instructions = QtWidgets.QLabel(
            self.tr(
                "The outer boundary follows the imaging circle. "
                "Adjust the inner-edge search position and smoothness, "
                "then confirm and click four points to cut."
            )
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self._preview = QtWidgets.QLabel()
        self._preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._preview.setFixedSize(720, 520)
        layout.addWidget(self._preview)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel(self.tr("Inner-edge position")))
        self._radius = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._radius.setRange(5, 60)
        self._radius.setValue(parameters.radius_percent)
        controls.addWidget(self._radius)
        self._value = QtWidgets.QLabel()
        controls.addWidget(self._value)
        auto = QtWidgets.QPushButton(self.tr("Reset"))
        auto.clicked.connect(lambda: self._radius.setValue(radius_percent))
        controls.addWidget(auto)
        layout.addLayout(controls)
        smoothing = QtWidgets.QHBoxLayout()
        smoothing.addWidget(QtWidgets.QLabel(self.tr("Contour smoothness")))
        self._smoothness = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._smoothness.setRange(1, 20)
        self._smoothness.setValue(parameters.smoothness)
        smoothing.addWidget(self._smoothness)
        layout.addLayout(smoothing)
        spacing = QtWidgets.QHBoxLayout()
        spacing.addWidget(QtWidgets.QLabel(self.tr("Point spacing")))
        self._point_spacing = QtWidgets.QSpinBox()
        self._point_spacing.setRange(2, 500)
        self._point_spacing.setValue(parameters.point_spacing)
        self._point_spacing.setSuffix(self.tr(" px"))
        self._point_spacing.setToolTip(
            self.tr("Larger spacing creates fewer polygon points")
        )
        spacing.addWidget(self._point_spacing)
        spacing.addStretch()
        layout.addLayout(spacing)
        self._status = QtWidgets.QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._refresh)
        self._radius.valueChanged.connect(self._schedule_refresh)
        self._smoothness.valueChanged.connect(self._schedule_refresh)
        self._refresh()

    @property
    def point_spacing(self) -> int:
        return self._point_spacing.value()

    def accept(self) -> None:
        if self._settings is not None:
            self._settings.setValue(RADIUS_SETTING, self._radius.value())
            self._settings.setValue(SMOOTHNESS_SETTING, self._smoothness.value())
            self._settings.setValue(POINT_SPACING_SETTING, self.point_spacing)
            self._settings.sync()
        super().accept()

    def _schedule_refresh(self, _value: int) -> None:
        self._value.setText(f"{self._radius.value()}%")
        self._buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            False
        )
        self._timer.start()

    def _refresh(self) -> None:
        self._value.setText(f"{self._radius.value()}%")
        preview = QtGui.QPixmap.fromImage(self._image).scaled(
            self._preview.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        try:
            if self._circle is None:
                raise ValueError("No imaging circle")
            self.mask = trace_imaging_ring(
                self._gray,
                circle=self._circle,
                inner_radius=self._circle[2] * self._radius.value() / 100,
                smoothness=self._smoothness.value() / 10,
            )
        except ValueError:
            self.mask = None
            self._status.setText(
                self.tr(
                    "Could not identify the imaging circle. "
                    "Use an image with a visible circular field on a dark background."
                )
            )
        else:
            painter = QtGui.QPainter(preview)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 100), 2))
            scale = preview.width() / self._image.width()
            for contour in mask_contours(self.mask):
                painter.drawPolyline(
                    QtGui.QPolygonF(
                        [
                            QtCore.QPointF(float(x * scale), float(y * scale))
                            for x, y in contour
                        ]
                    )
                )
            painter.end()
            self._status.setText(
                self.tr("Green lines show the extracted outer and inner boundaries.")
            )
        self._preview.setPixmap(preview)
        self._buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.mask is not None
        )
