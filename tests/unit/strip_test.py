from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from labelme._strip import DEFAULT_HALF_WIDTH
from labelme._strip import centerline_to_strip
from labelme._strip import load_strip_half_width
from labelme._strip import save_strip_half_width


def test_centerline_to_strip_creates_closed_width_around_line() -> None:
    polygon = centerline_to_strip(
        [[20, 50], [80, 50]], half_width=5, image_shape=(100, 100)
    )

    assert len(polygon) >= 4
    assert polygon[:, 0].min() == pytest.approx(14.5, abs=1)
    assert polygon[:, 0].max() == pytest.approx(85.5, abs=1)
    assert polygon[:, 1].min() == pytest.approx(44.5, abs=1)
    assert polygon[:, 1].max() == pytest.approx(55.5, abs=1)


def test_centerline_to_strip_follows_bent_centerline() -> None:
    polygon = centerline_to_strip(
        [[20, 20], [50, 20], [50, 70]],
        half_width=6,
        image_shape=(100, 100),
    )

    assert np.any(np.linalg.norm(polygon - [20, 20], axis=1) <= 7)
    assert np.any(np.linalg.norm(polygon - [50, 70], axis=1) <= 7)
    assert polygon[:, 0].max() >= 55


def test_strip_half_width_can_be_loaded_from_ini(tmp_path: Path) -> None:
    config_file = tmp_path / "config.ini"
    config_file.write_text("[strip]\nhalf_width = 12\n", encoding="utf-8")

    assert load_strip_half_width(config_file=config_file) == 12


@pytest.mark.parametrize("value", [0, 101, "wide"])
def test_invalid_strip_half_width_uses_default(
    tmp_path: Path, value: int | str
) -> None:
    config_file = tmp_path / "config.ini"
    config_file.write_text(
        f"[strip]\nhalf_width = {value}\n", encoding="utf-8"
    )

    assert load_strip_half_width(config_file=config_file) == DEFAULT_HALF_WIDTH


def test_strip_half_width_can_be_saved_without_losing_comments(tmp_path: Path) -> None:
    config_file = tmp_path / "config.ini"
    config_file.write_text(
        "[ring]\n# Keep this comment.\npoint_spacing = 36\n\n"
        "[strip]\n# Keep this too.\nhalf_width = 4\n",
        encoding="utf-8",
    )

    save_strip_half_width(15, config_file=config_file)

    assert load_strip_half_width(config_file=config_file) == 15
    assert "# Keep this comment." in config_file.read_text(encoding="utf-8")
    assert "# Keep this too." in config_file.read_text(encoding="utf-8")
