from __future__ import annotations

from pathlib import Path

from labelme._roi_tools_config import RoiToolConfig
from labelme._roi_tools_config import load_roi_tool_config
from labelme._roi_tools_config import merge_roi_shortcuts


def test_load_roi_tool_config_from_ini(tmp_path: Path) -> None:
    config_file = tmp_path / "config.ini"
    config_file.write_text(
        "[labels]\npolygon = a\nstrip = b\nring = c\n"
        "[shortcuts]\ncreate_polygon = P\ncreate_strip = S\ncreate_ring = R\n",
        encoding="utf-8",
    )

    config = load_roi_tool_config(config_file=config_file)

    assert config == RoiToolConfig("a", "b", "c", "P", "S", "R")


def test_roi_shortcuts_replace_conflicting_builtin_shortcuts() -> None:
    merged = merge_roi_shortcuts(
        {
            "quit": "Ctrl+Q",
            "close": "Ctrl+W",
            "create_polygon": "Ctrl+N",
            "open_next": ["D", "Ctrl+E"],
        },
        roi=RoiToolConfig(),
    )

    assert merged["quit"] is None
    assert merged["close"] is None
    assert merged["open_next"] == ["D"]
    assert merged["create_polygon"] == "Ctrl+E"
    assert merged["create_strip"] == "Ctrl+Q"
    assert merged["create_annular_sector"] == "Ctrl+W"
