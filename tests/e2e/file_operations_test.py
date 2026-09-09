from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6 import QtCore
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from labelme._app import MainWindow

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import select_shape
from .conftest import show_window_and_wait_for_imagedata


@pytest.mark.gui
@pytest.mark.parametrize(
    "set_fit_mode",
    [MainWindow.set_fit_window_mode, MainWindow.set_fit_width_mode],
)
def test_close_file(
    *,
    annotated_win: MainWindow,
    qtbot: QtBot,
    pause: bool,
    set_fit_mode: Callable[[MainWindow], None],
) -> None:
    assert annotated_win._annotation is not None
    assert annotated_win._canvas_widgets.canvas.isEnabled()

    set_fit_mode(annotated_win)
    annotated_win.close_file()
    annotated_win.resize(annotated_win.width() + 50, annotated_win.height() + 50)
    qtbot.wait(50)

    assert not annotated_win._canvas_widgets.canvas.isEnabled()
    assert annotated_win._annotation is None
    assert annotated_win.windowTitle() == "Labelme"

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_delete_label_file(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    pause: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    win = main_win(
        file_or_dir=str(data_path / "annotated"),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    label_file = data_path / "annotated/2011_000003.json"
    assert label_file.exists()

    item = win._docks.file_list.currentItem()
    assert item is not None
    assert item.checkState() == Qt.CheckState.Checked

    monkeypatch.setattr(win, "_confirm_deletion", lambda *_args, **_kwargs: True)
    win.delete_file()
    qtbot.wait(50)

    assert not label_file.exists()
    assert win._label_file_path is None

    item = win._docks.file_list.currentItem()
    assert item is not None
    assert item.checkState() == Qt.CheckState.Unchecked

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_ctrl_selected_label_files_are_deleted_together(
    *,
    main_win: MainWinFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    win = main_win()
    image_paths = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    label_paths = [path.with_suffix(".json") for path in image_paths]
    for label_path in label_paths:
        label_path.write_text("{}")

    win._loaded_image_paths = [str(path) for path in image_paths]
    win._refresh_file_list()
    assert win._status_bar.annotation_progress.text() == "Labeled: 2/2"
    with QtCore.QSignalBlocker(win._docks.file_list):
        for row in range(win._docks.file_list.count()):
            win._docks.file_list.item(row).setSelected(True)
    win._update_delete_file_action()

    confirmations: list[tuple[str, bool]] = []

    def confirm(*, message: str, default_delete: bool = False) -> bool:
        confirmations.append((message, default_delete))
        return True

    monkeypatch.setattr(win, "_confirm_deletion", confirm)
    win.delete_file()

    remaining = [path for path in label_paths if path.exists()]
    assert win._status_bar.annotation_progress.text() == "Labeled: 0/2"
    win.close()
    assert remaining == []
    assert confirmations and "2" in confirmations[0][0]
    assert confirmations[0][1] is True


@pytest.mark.gui
def test_delete_key_in_file_list_deletes_label_file(
    *,
    qtbot: QtBot,
    main_win: MainWinFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    win = main_win()
    image_path = tmp_path / "image.jpg"
    label_path = image_path.with_suffix(".json")
    label_path.write_text("{}")
    win._loaded_image_paths = [str(image_path)]
    win._refresh_file_list()
    with QtCore.QSignalBlocker(win._docks.file_list):
        win._docks.file_list.setCurrentRow(0)
    win._update_delete_file_action()
    monkeypatch.setattr(win, "_confirm_deletion", lambda **_kwargs: True)

    qtbot.keyClick(win._docks.file_list, Qt.Key.Key_Delete)

    exists_after_delete = label_path.exists()
    win.close()
    assert not exists_after_delete


@pytest.mark.gui
def test_delete_label_file_keeps_image(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    pause: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    win = main_win(
        file_or_dir=str(data_path / "annotated"),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    canvas = win._canvas_widgets.canvas
    assert not canvas.pixmap.isNull()
    assert canvas.shapes
    assert len(win._docks.label_list) > 0

    monkeypatch.setattr(win, "_confirm_deletion", lambda *_args, **_kwargs: True)
    win.delete_file()
    qtbot.wait(50)

    # The annotations are cleared, but the image stays on the canvas.
    assert not canvas.pixmap.isNull()
    assert canvas.isEnabled()
    assert canvas.shapes == []
    assert len(win._docks.label_list) == 0
    assert win._image_path is not None
    assert win._annotation is not None

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_delete_file_respects_output_dir(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    pause: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    win = main_win(
        file_or_dir=str(data_path / "raw/2011_000003.jpg"),
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    saved_path = tmp_path / "2011_000003.json"
    image_adjacent_path = data_path / "raw/2011_000003.json"
    assert not saved_path.exists()
    assert not image_adjacent_path.exists()

    # Before any save, no label file is tracked, so the prospective path must
    # still resolve under the output dir rather than next to the image.
    assert win.current_label_file_path() == str(saved_path)
    assert not win.has_label_file()

    win.save_labels(label_path=str(saved_path))
    assert saved_path.exists()

    assert win.current_label_file_path() == str(saved_path)
    assert win.has_label_file()

    monkeypatch.setattr(win, "_confirm_deletion", lambda *_args, **_kwargs: True)
    win.delete_file()
    qtbot.wait(50)

    assert not saved_path.exists()
    assert not image_adjacent_path.exists()

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_current_label_file_path_prefers_opened_file(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    pause: bool,
) -> None:
    opened_path = data_path / "annotated/2011_000003.json"
    win = main_win(
        file_or_dir=str(opened_path),
        output_dir=str(tmp_path),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    assert win.current_label_file_path() == str(opened_path)
    assert win.has_label_file()
    assert not (tmp_path / "2011_000003.json").exists()

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


@pytest.mark.gui
def test_undo_after_delete_file_does_not_restore_shapes(
    *,
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    pause: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    win = main_win(
        file_or_dir=str(data_path / "annotated"),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    canvas = win._canvas_widgets.canvas
    monkeypatch.setattr(win, "_confirm_deletion", lambda *_args, **_kwargs: True)

    # A prior shape edit in the same session enables the undo action.
    win._switch_canvas_mode(edit=True, create_mode=None)
    select_shape(qtbot=qtbot, canvas=canvas, shape_index=0)
    win.delete_selected_shapes()
    qtbot.wait(50)
    assert canvas.can_restore_shape
    assert win._actions.undo.isEnabled()

    win.delete_file()
    qtbot.wait(50)
    assert canvas.shapes == []

    # Undo must not resurrect the annotations of the file removed from disk.
    assert not canvas.can_restore_shape
    assert not win._actions.undo.isEnabled()
    win.undo_shape_edit()
    assert canvas.shapes == []
    assert len(win._docks.label_list) == 0

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)
