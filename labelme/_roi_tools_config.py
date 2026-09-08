from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ._ring_config import get_ring_config_file


@dataclass(frozen=True)
class RoiToolConfig:
    polygon_label: str = "1"
    strip_label: str = "2"
    ring_label: str = "5"
    polygon_shortcut: str = "Ctrl+E"
    strip_shortcut: str = "Ctrl+Q"
    ring_shortcut: str = "Ctrl+W"


def load_roi_tool_config(*, config_file: Path | None = None) -> RoiToolConfig:
    config_file = config_file or get_ring_config_file()
    parser = configparser.ConfigParser()
    try:
        if config_file.is_file():
            parser.read(config_file, encoding="utf-8")
        defaults = RoiToolConfig()
        values = {
            "polygon_label": parser.get(
                "labels", "polygon", fallback=defaults.polygon_label
            ),
            "strip_label": parser.get("labels", "strip", fallback=defaults.strip_label),
            "ring_label": parser.get("labels", "ring", fallback=defaults.ring_label),
            "polygon_shortcut": parser.get(
                "shortcuts", "create_polygon", fallback=defaults.polygon_shortcut
            ),
            "strip_shortcut": parser.get(
                "shortcuts", "create_strip", fallback=defaults.strip_shortcut
            ),
            "ring_shortcut": parser.get(
                "shortcuts", "create_ring", fallback=defaults.ring_shortcut
            ),
        }
    except (configparser.Error, OSError) as error:
        logger.warning(
            "Cannot read ROI tool settings from {!r}: {}", config_file, error
        )
        return RoiToolConfig()
    return RoiToolConfig(**{key: value.strip() for key, value in values.items()})


def merge_roi_shortcuts(
    shortcuts: dict[str, Any], *, roi: RoiToolConfig
) -> dict[str, Any]:
    """Apply INI shortcuts and remove collisions with built-in actions."""
    result = dict(shortcuts)
    custom = {
        "create_polygon": roi.polygon_shortcut,
        "create_annular_sector": roi.ring_shortcut,
        "create_strip": roi.strip_shortcut,
    }
    reserved = {value.casefold() for value in custom.values() if value}
    for key, value in tuple(result.items()):
        if key in custom:
            continue
        if isinstance(value, list):
            result[key] = [
                item
                for item in value
                if not isinstance(item, str) or item.casefold() not in reserved
            ]
        elif isinstance(value, str) and value.casefold() in reserved:
            result[key] = None
    result.update(custom)
    return result
