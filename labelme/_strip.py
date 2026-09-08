from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy import ndimage
from skimage import draw
from skimage import measure

from ._ring_config import get_ring_config_file

DEFAULT_HALF_WIDTH = 8
MINIMUM_HALF_WIDTH = 1
MAXIMUM_HALF_WIDTH = 100
COORDINATE_COUNT = 2
MINIMUM_CENTERLINE_POINTS = 2
MINIMUM_POLYGON_POINTS = 3


def load_strip_half_width(*, config_file: Path | None = None) -> int:
    """Read the strip half-width from the shared project INI file."""
    config_file = config_file or get_ring_config_file()
    parser = configparser.ConfigParser()
    try:
        if not config_file.is_file():
            return DEFAULT_HALF_WIDTH
        parser.read(config_file, encoding="utf-8")
        value = parser.getint("strip", "half_width")
        if not MINIMUM_HALF_WIDTH <= value <= MAXIMUM_HALF_WIDTH:
            raise ValueError(
                f"half_width must be between {MINIMUM_HALF_WIDTH} "
                f"and {MAXIMUM_HALF_WIDTH}"
            )
    except (configparser.Error, OSError, ValueError) as error:
        logger.warning(
            "Cannot read strip half-width from {!r}: {}; using {}",
            str(config_file),
            error,
            DEFAULT_HALF_WIDTH,
        )
        return DEFAULT_HALF_WIDTH
    return value


def save_strip_half_width(
    value: int, *, config_file: Path | None = None
) -> None:
    """Persist the strip half-width while preserving existing INI comments."""
    if not MINIMUM_HALF_WIDTH <= value <= MAXIMUM_HALF_WIDTH:
        raise ValueError(
            f"half_width must be between {MINIMUM_HALF_WIDTH} "
            f"and {MAXIMUM_HALF_WIDTH}"
        )
    config_file = config_file or get_ring_config_file()
    content = config_file.read_text(encoding="utf-8") if config_file.exists() else ""
    lines = content.splitlines()
    section_index: int | None = None
    next_section_index = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.casefold() == "[strip]":
            section_index = index
            continue
        if section_index is not None and stripped.startswith("["):
            next_section_index = index
            break
    setting = f"half_width = {value}"
    if section_index is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(("[strip]", setting))
    else:
        for index in range(section_index + 1, next_section_index):
            if lines[index].partition("=")[0].strip().casefold() == "half_width":
                lines[index] = setting
                break
        else:
            lines.insert(next_section_index, setting)
    config_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def centerline_to_strip(
    points: npt.ArrayLike,
    *,
    half_width: float,
    image_shape: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Expand a clicked centerline into a clipped, closed strip polygon."""
    centerline = np.asarray(points, dtype=np.float64)
    if (
        centerline.ndim != COORDINATE_COUNT
        or centerline.shape[1] != COORDINATE_COUNT
        or len(centerline) < MINIMUM_CENTERLINE_POINTS
    ):
        raise ValueError("A strip needs at least two centerline points")
    if not np.isfinite(centerline).all() or not np.isfinite(half_width):
        raise ValueError("Strip coordinates and width must be finite")
    if half_width <= 0:
        raise ValueError("Strip width must be positive")
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")

    line_mask = np.zeros((height, width), dtype=np.bool_)
    rounded = np.rint(centerline).astype(int)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    for start, end in zip(rounded[:-1], rounded[1:]):
        rows, columns = draw.line(start[1], start[0], end[1], end[0])
        line_mask[rows, columns] = True
    if np.count_nonzero(line_mask) < MINIMUM_CENTERLINE_POINTS:
        raise ValueError("Strip centerline has zero length")

    radius = int(np.ceil(half_width))
    row_offsets, column_offsets = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = row_offsets**2 + column_offsets**2 <= half_width**2
    strip_mask = ndimage.binary_dilation(line_mask, structure=disk)
    contours = measure.find_contours(np.pad(strip_mask, 1), 0.5)
    if not contours:
        raise ValueError("Could not create strip contour")
    contour = max(contours, key=len)[:, ::-1] - 1
    polygon = _simplify_closed_contour(contour, tolerance=1.0)
    if np.array_equal(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    if len(polygon) < MINIMUM_POLYGON_POINTS:
        raise ValueError("Strip polygon is degenerate")
    return polygon


def _simplify_open_contour(
    points: npt.NDArray[np.float64], *, tolerance: float
) -> npt.NDArray[np.float64]:
    """Simplify an open contour with the Ramer-Douglas-Peucker algorithm."""
    if len(points) <= MINIMUM_CENTERLINE_POINTS:
        return points
    segment = points[-1] - points[0]
    length = float(np.linalg.norm(segment))
    if length == 0:
        distances = np.linalg.norm(points - points[0], axis=1)
    else:
        offsets = points - points[0]
        distances = np.abs(segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0])
        distances /= length
    split = int(np.argmax(distances))
    if distances[split] <= tolerance:
        return points[[0, -1]]
    first = _simplify_open_contour(points[: split + 1], tolerance=tolerance)
    second = _simplify_open_contour(points[split:], tolerance=tolerance)
    return np.vstack((first[:-1], second))


def _simplify_closed_contour(
    points: npt.NDArray[np.float64], *, tolerance: float
) -> npt.NDArray[np.float64]:
    """Simplify a closed contour by splitting it at a distant vertex."""
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    split = int(np.linalg.norm(points - points[0], axis=1).argmax())
    first = _simplify_open_contour(points[: split + 1], tolerance=tolerance)
    second_path = np.vstack((points[split:], points[0]))
    second = _simplify_open_contour(second_path, tolerance=tolerance)
    return np.vstack((first[:-1], second[:-1]))
