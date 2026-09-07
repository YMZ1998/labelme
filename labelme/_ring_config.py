from __future__ import annotations

import configparser
import os
from pathlib import Path

from loguru import logger

DEFAULT_POINT_SPACING = 24
MINIMUM_POINT_SPACING = 2
MAXIMUM_POINT_SPACING = 500


def get_ring_config_file() -> Path:
    """Return the ring INI file, allowing launchers to override its location."""
    if configured := os.environ.get("LABELME_CONFIG_INI"):
        return Path(configured)
    working_copy = Path.cwd() / "config.ini"
    if working_copy.exists():
        return working_copy
    return Path(__file__).resolve().parent.parent / "config.ini"


def load_ring_point_spacing(*, config_file: Path | None = None) -> int:
    """Read the curved-boundary point spacing, falling back on invalid input."""
    config_file = config_file or get_ring_config_file()
    parser = configparser.ConfigParser()
    try:
        if not config_file.is_file():
            return DEFAULT_POINT_SPACING
        parser.read(config_file, encoding="utf-8")
        value = parser.getint("ring", "point_spacing")
        if not MINIMUM_POINT_SPACING <= value <= MAXIMUM_POINT_SPACING:
            raise ValueError(
                f"point_spacing must be between {MINIMUM_POINT_SPACING} "
                f"and {MAXIMUM_POINT_SPACING}"
            )
    except (configparser.Error, OSError, ValueError) as error:
        logger.warning(
            "Cannot read ring point spacing from {!r}: {}; using {}",
            str(config_file),
            error,
            DEFAULT_POINT_SPACING,
        )
        return DEFAULT_POINT_SPACING
    return value
