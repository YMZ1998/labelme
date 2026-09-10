from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from PySide6 import QtGui
from pytestqt.qtbot import QtBot

from labelme._ring_config import DEFAULT_POINT_SPACING
from labelme._ring_config import load_ring_point_spacing
from labelme._ring_segmentation import cut_ring
from labelme._ring_segmentation import fit_imaging_circle
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
        & np.isclose(polygon[:, 1], 100)
    ]
    second_side = polygon[
        (polygon[:, 1] >= 119.5)
        & (polygon[:, 1] <= 180.5)
        & np.isclose(polygon[:, 0], 100)
    ]
    assert len(first_side) == 3
    assert len(second_side) == 3
    assert np.any(np.all(np.isclose(first_side, [150, 100]), axis=1))
    assert np.any(np.all(np.isclose(second_side, [100, 150]), axis=1))


def test_parallel_cutting_sides_have_no_angle_restriction() -> None:
    yy, xx = np.indices((201, 201))
    radius = np.hypot(xx - 100, yy - 100)
    mask = (radius >= 20) & (radius <= 80)
    parallel_controls = [[100, 20], [100, 80], [100, 120], [100, 180]]

    first_side = cut_ring(mask, parallel_controls, major_arc=True)
    other_side = cut_ring(mask, parallel_controls, major_arc=False)

    assert len(first_side) >= 3
    assert len(other_side) >= 3
    assert np.mean(first_side[:, 0]) != pytest.approx(np.mean(other_side[:, 0]))


def test_independently_angled_sides_do_not_need_to_meet_in_hole() -> None:
    yy, xx = np.indices((201, 201))
    radius = np.hypot(xx - 100, yy - 100)
    mask = (radius >= 20) & (radius <= 80)
    angles_and_radii = [(270, 80), (250, 20), (30, 20), (50, 80)]
    controls = np.array(
        [
            [
                100 + radius * np.cos(np.deg2rad(angle)),
                100 + radius * np.sin(np.deg2rad(angle)),
            ]
            for angle, radius in angles_and_radii
        ]
    )

    polygon = cut_ring(mask, controls, major_arc=True)

    for control in controls:
        assert np.min(np.linalg.norm(polygon - control, axis=1)) < 1e-6


def test_nearly_collinear_sides_tolerate_points_inside_boundaries() -> None:
    yy, xx = np.indices((401, 401))
    radius = np.hypot(xx - 200, yy - 200)
    mask = (radius >= 70) & (radius <= 180)
    controls = [[196, 31], [199, 139], [205, 261], [214, 369]]

    polygon = cut_ring(mask, controls, major_arc=True, point_spacing=20)

    assert len(polygon) >= 4
    assert np.isfinite(polygon).all()


def test_dialog_defaults_to_lower_density(qtbot: QtBot) -> None:
    image = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor("black"))
    dialog = RingContourDialog(image=image, parent=None)
    qtbot.addWidget(dialog)
    assert dialog.point_spacing == load_ring_point_spacing()
    dialog._point_spacing.setValue(24)
    assert dialog.point_spacing == 24


def test_dialog_remembers_accepted_parameters(qtbot: QtBot, tmp_path: Path) -> None:
    image = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor("black"))
    settings = QtCore.QSettings(
        str(tmp_path / "ring-settings.ini"), QtCore.QSettings.Format.IniFormat
    )
    dialog = RingContourDialog(image=image, parent=None, settings=settings)
    qtbot.addWidget(dialog)
    dialog._radius.setValue(37)
    dialog._smoothness.setValue(12)
    dialog._point_spacing.setValue(36)
    dialog.accept()

    restored = RingContourDialog(image=image, parent=None, settings=settings)
    qtbot.addWidget(restored)

    assert restored._radius.value() == 37
    assert restored._smoothness.value() == 12
    assert restored.point_spacing == 36


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


def test_outer_circle_uses_low_contrast_outer_edge() -> None:
    yy, xx = np.indices((240, 240))
    radius = np.hypot(xx - 120, yy - 120)
    rng = np.random.default_rng(7)
    image = np.zeros((240, 240), dtype=np.float64)
    interior = radius <= 88
    image[interior] = 35 + rng.normal(0, 8, np.count_nonzero(interior))
    transition = (radius > 88) & (radius <= 100)
    image[transition] = (
        (100 - radius[transition]) / 12 * 35
        + rng.normal(0, 3, np.count_nonzero(transition))
    )
    image = np.clip(image, 0, 255)

    center_x, center_y, fitted_radius = fit_imaging_circle(image)

    assert center_x == pytest.approx(120, abs=2)
    assert center_y == pytest.approx(120, abs=2)
    assert fitted_radius == pytest.approx(94, abs=4)
