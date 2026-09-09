from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6 import QtCore
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from labelme._onnx_segmentation import OnnxPrediction
from labelme._shape import Shape

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata


def test_onnx_prediction_action_adds_editable_shapes(
    *,
    qtbot: QtBot,
    main_win: MainWinFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pause: bool,
) -> None:
    image_path = tmp_path / "oct.png"
    Image.fromarray(np.full((64, 64), 127, dtype=np.uint8)).save(image_path)
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    win = main_win(
        file_or_dir=image_path,
        config_overrides={"auto_save": False},
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    assert not win._actions.reload_dir.icon().isNull()
    reload_buttons = [
        button
        for button in win.findChildren(QtWidgets.QToolButton)
        if button.defaultAction() is win._actions.reload_dir
    ]
    assert len(reload_buttons) == 1
    assert (
        reload_buttons[0].toolButtonStyle()
        == QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
    )
    assert not win._ai_annotation.isVisible()
    assert not win._ai_text.isVisible()
    predicted = Shape(
        label="1",
        shape_type="polygon",
        points=np.array([[10, 10], [30, 10], [20, 30]], dtype=np.float64),
        closed=True,
    )
    monkeypatch.setattr(win, "_onnx_model_path", lambda: model_path)
    predict_kwargs: dict[str, object] = {}

    def predict(**kwargs: object) -> OnnxPrediction:
        predict_kwargs.update(kwargs)
        return OnnxPrediction(
            shapes=[predicted],
            provider="CPUExecutionProvider",
            elapsed_ms=12.5,
            model_path=model_path,
        )

    monkeypatch.setattr(
        win._onnx_segmenter,
        "predict",
        predict,
    )

    assert win._actions.onnx_predict.isEnabled()
    assert win._actions.onnx_predict.shortcut().toString() == "Shift+A"
    win._actions.onnx_predict.trigger()

    assert predicted in win._canvas_widgets.canvas.shapes
    assert win._docks.label_list.find_item_by_shape(shape=predicted) is not None
    assert win._is_changed
    assert predict_kwargs["keep_classes"] == ["1", "2", "3"]
    assert predict_kwargs["minimum_polygon_area"] == 0
    win.mark_clean()
    close_or_pause(qtbot=qtbot, widget=win, pause=pause)
