from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtGui
from pytestqt.qtbot import QtBot

from labelme._ring_config import DEFAULT_POINT_SPACING
from labelme._ring_config import load_ring_point_spacing
from labelme._ring_segmentation import cut_ring
from labelme._ring_segmentation import resample_closed_contour
from labelme._ring_segmentation import trace_default_imaging_ring
from labelme._widgets.ring_contour_dialog import RingContourDialog


def test_resample_closed_contour_uses_requested_spacing() -> None:
    contour = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]])
    sampled = resample_closed_contour(contour, point_spacing=20)
    assert len(sampled) == 20
    distances = np.linalg.norm(np.roll(sampled, -1, axis=0) - sampled, axis=1)
    np.testing.assert_allclose(distances, 20)


def test_larger_spacing_generates_fewer_ring_points() -> None:
    yy, xx = np.indices((201, 201))
    radius = np.hypot(xx - 100, yy - 100)
    mask = (radius >= 20) & (radius <= 80)
    controls = [[180, 100], [120, 100], [100, 120], [100, 180]]
    dense = cut_ring(mask, controls, point_spacing=4)
    sparse = cut_ring(mask, controls, point_spacing=16)
    assert len(sparse) < len(dense) / 3
    fixed_side_points = 4
    expected = (len(dense) - fixed_side_points) / 4 + fixed_side_points
    assert len(sparse) == pytest.approx(expected, abs=2)


def test_cutting_sides_each_have_only_three_points() -> None:
    yy, xx = np.indices((201, 201))
    radius = np.hypot(xx - 100, yy - 100)
    mask = (radius >= 20) & (radius <= 80)
    controls = [[180, 100], [120, 100], [100, 120], [100, 180]]

    polygon = cut_ring(mask, controls, point_spacing=4)

    first_side = polygon[
        (polygon[:, 0] >= 120)
        & (polygon[:, 0] <= 180)
        & (np.abs(polygon[:, 1] - 100) <= 0.5)
    ]
    second_side = polygon[
        (polygon[:, 1] >= 119.5)
        & (polygon[:, 1] <= 180.5)
        & (np.abs(polygon[:, 0] - 100) <= 0.5)
    ]
    assert len(first_side) == 3
    assert len(second_side) == 3
    np.testing.assert_allclose(first_side[1], [150, 100])
    np.testing.assert_allclose(second_side[-1], [100, 150])


def test_dialog_defaults_to_lower_density(qtbot: QtBot) -> None:
    image = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor("black"))
    dialog = RingContourDialog(image=image, parent=None)
    qtbot.addWidget(dialog)
    assert dialog.point_spacing == load_ring_point_spacing()
    dialog._point_spacing.setValue(24)
    assert dialog.point_spacing == 24


def test_point_spacing_can_be_loaded_from_ini(tmp_path: Path) -> None:
    config_file = tmp_path / "config.ini"
    config_file.write_text("[ring]\npoint_spacing = 36\n", encoding="utf-8")

    assert load_ring_point_spacing(config_file=config_file) == 36


@pytest.mark.parametrize("value", [0, 501, "many"])
def test_invalid_ini_point_spacing_uses_default(
    tmp_path: Path, value: int | str
) -> None:
    config_file = tmp_path / "config.ini"
    config_file.write_text(f"[ring]\npoint_spacing = {value}\n", encoding="utf-8")

    assert load_ring_point_spacing(config_file=config_file) == DEFAULT_POINT_SPACING


def test_default_ring_extraction_finds_annular_region() -> None:
    yy, xx = np.indices((201, 201))
    radius = np.hypot(xx - 100, yy - 100)
    image = np.zeros((201, 201, 3), dtype=np.uint8)
    image[(radius >= 30) & (radius <= 80)] = 180

    mask = trace_default_imaging_ring(image)

    assert mask[100, 150]
    assert not mask[100, 100]
    assert not mask[100, 190]
