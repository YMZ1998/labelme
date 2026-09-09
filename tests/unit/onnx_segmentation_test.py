import os
from pathlib import Path

import numpy as np
import pytest

from labelme._onnx_segmentation import OnnxSegmenter
from labelme._onnx_segmentation import discover_onnx_model
from labelme._onnx_segmentation import mask_oct_region
from labelme._onnx_segmentation import onnx_output_to_mask
from labelme._onnx_segmentation import prepare_onnx_input
from labelme._onnx_segmentation import shapes_from_onnx_mask


def test_predict_rejects_unknown_keep_class(tmp_path: Path) -> None:
    segmenter = OnnxSegmenter()

    with pytest.raises(ValueError, match="Unknown ONNX classes"):
        segmenter.predict(
            image=np.zeros((8, 8), dtype=np.uint8),
            model_path=tmp_path / "missing.onnx",
            polygon_detail=50,
            keep_classes=["4"],
        )


def test_shapes_from_onnx_mask_have_no_group_and_filter_classes() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:20, 5:20] = 1
    mask[35:55, 35:55] = 1
    mask[20:40, 20:40] = 2

    shapes = shapes_from_onnx_mask(
        mask=mask,
        model_name="model.onnx",
        image_size=(64, 64),
        polygon_detail=50,
        keep_classes=["1"],
    )

    assert len(shapes) == 2
    assert {shape.label for shape in shapes} == {"1"}
    assert all(shape.group_id is None for shape in shapes)


def test_shapes_from_onnx_mask_filters_small_polygons_by_area() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[2:7, 2:7] = 1
    mask[20:50, 20:50] = 1

    shapes = shapes_from_onnx_mask(
        mask=mask,
        model_name="model.onnx",
        image_size=(64, 64),
        polygon_detail=100,
        keep_classes=["1"],
        minimum_polygon_area=100,
    )

    assert len(shapes) == 1
    assert shapes[0].points[:, 0].max() > 40


def test_prepare_onnx_input_supports_nchw_and_nhwc() -> None:
    gray = np.arange(24, dtype=np.uint8).reshape(4, 6)

    nchw = prepare_onnx_input(gray, [1, 3, 8, 10])
    nhwc = prepare_onnx_input(gray, [1, 8, 10, 1])

    assert nchw.shape == (1, 3, 8, 10)
    assert nhwc.shape == (1, 8, 10, 1)
    assert nchw.dtype == np.float32
    assert nhwc.dtype == np.float32
    assert -1 <= nchw.min() <= nchw.max() <= 1


def test_onnx_output_to_mask_accepts_logits_and_scaled_masks() -> None:
    logits = np.zeros((1, 4, 2, 2), dtype=np.float32)
    logits[0, 1, 0, 0] = 1
    logits[0, 2, 0, 1] = 1
    logits[0, 3, 1, 0] = 1

    assert onnx_output_to_mask(logits).tolist() == [[1, 2], [3, 0]]
    assert onnx_output_to_mask(np.array([[0, 64, 128, 192]])).tolist() == [
        [0, 1, 2, 3]
    ]


def test_onnx_output_to_mask_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unknown predicted class values"):
        onnx_output_to_mask(np.array([[17]], dtype=np.uint8))


def test_mask_oct_region_zeros_corners() -> None:
    image = np.ones((20, 20), dtype=np.uint8)

    masked = mask_oct_region(image)

    assert masked[0, 0] == 0
    assert masked[10, 10] == 1


def test_discover_onnx_model_uses_newest_candidate(tmp_path: Path) -> None:
    repository_root = tmp_path / "labelme"
    first_dir = repository_root / "save_weights"
    second_dir = tmp_path / "2D-Image-Segmentation" / "save_weights"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True, exist_ok=True)
    older = first_dir / "older.onnx"
    newer = second_dir / "newer.onnx"
    older.touch()
    newer.touch()
    older.touch()
    newer.touch()
    older_time = older.stat().st_mtime - 10
    os.utime(older, (older_time, older_time))

    assert discover_onnx_model(repository_root=repository_root) == newer
