from __future__ import annotations

import dataclasses
import time
from collections.abc import Collection
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from PIL import Image

from ._automation._shape_builders import Detection
from ._automation._shape_builders import shapes_from_detections
from ._shape import Shape

DEFAULT_IMAGE_SIZE: Final[int] = 1024
DEFAULT_POLYGON_DETAIL: Final[int] = 50
OCT_OUTER_RADIUS_RATIO: Final[float] = 0.49
GRAYSCALE_DIMENSIONS: Final[int] = 2
ONNX_INPUT_DIMENSIONS: Final[int] = 4
CLASS_NAMES: Final[tuple[str, ...]] = (
    "background",
    "1",
    "2",
    "3",
)
MASK_VALUE_TO_CLASS_ID: Final[dict[int, int]] = {
    0: 0,
    1: 1,
    64: 1,
    2: 2,
    128: 2,
    3: 3,
    192: 3,
    255: 3,
}


@dataclasses.dataclass(frozen=True)
class OnnxPrediction:
    shapes: list[Shape]
    provider: str
    elapsed_ms: float
    model_path: Path


def discover_onnx_model(*, repository_root: Path) -> Path | None:
    directories = (
        repository_root / "save_weights",
        repository_root.parent / "2D-Image-Segmentation" / "save_weights",
    )
    candidates = [
        path for directory in directories for path in directory.glob("*.onnx")
    ]
    return (
        max(candidates, key=lambda path: path.stat().st_mtime)
        if candidates
        else None
    )


def mask_oct_region(
    image: npt.NDArray[np.generic], *, radius_ratio: float = OCT_OUTER_RADIUS_RATIO
) -> npt.NDArray[np.generic]:
    if image.ndim != GRAYSCALE_DIMENSIONS:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")
    height, width = image.shape
    yy, xx = np.ogrid[:height, :width]
    radius = min(height, width) * radius_ratio
    valid = (xx - width / 2.0) ** 2 + (yy - height / 2.0) ** 2 <= radius**2
    result = np.zeros_like(image)
    result[valid] = image[valid]
    return result


def prepare_onnx_input(
    gray: npt.NDArray[np.uint8],
    input_shape: list[int | str | None],
    *,
    fallback_size: int = DEFAULT_IMAGE_SIZE,
) -> npt.NDArray[np.float32]:
    if len(input_shape) != ONNX_INPUT_DIMENSIONS:
        raise ValueError(f"Expected a 4D ONNX input, got {input_shape}")

    def fixed(value: int | str | None) -> int:
        return value if isinstance(value, int) and value > 0 else fallback_size

    if input_shape[1] in (1, 3):
        layout, channels = "NCHW", int(input_shape[1])
        height, width = fixed(input_shape[2]), fixed(input_shape[3])
    elif input_shape[3] in (1, 3):
        layout, channels = "NHWC", int(input_shape[3])
        height, width = fixed(input_shape[1]), fixed(input_shape[2])
    else:
        raise ValueError(f"Cannot determine ONNX input layout from {input_shape}")

    resized = np.asarray(
        Image.fromarray(gray).resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    normalized = resized / 127.5 - 1.0
    channels_last = np.repeat(normalized[..., None], channels, axis=2)
    tensor = (
        channels_last.transpose(2, 0, 1)[None]
        if layout == "NCHW"
        else channels_last[None]
    )
    return np.ascontiguousarray(tensor, dtype=np.float32)


def onnx_output_to_mask(output: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    array = np.asarray(output)
    if array.ndim == ONNX_INPUT_DIMENSIONS:
        valid_counts = range(2, len(CLASS_NAMES) + 1)
        if array.shape[1] in valid_counts:
            array = array.argmax(axis=1)[0]
        elif array.shape[-1] in valid_counts:
            array = array.argmax(axis=-1)[0]
        else:
            raise ValueError(f"Cannot find class axis in ONNX output {array.shape}")
    elif array.ndim == GRAYSCALE_DIMENSIONS + 1 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim != GRAYSCALE_DIMENSIONS:
        raise ValueError(f"Unsupported ONNX output shape {array.shape}")

    raw = array.astype(np.uint8)
    mapped = np.zeros_like(raw)
    unknown = set(np.unique(raw).tolist()) - MASK_VALUE_TO_CLASS_ID.keys()
    if unknown:
        raise ValueError(f"Unknown predicted class values: {sorted(unknown)}")
    for value, class_id in MASK_VALUE_TO_CLASS_ID.items():
        mapped[raw == value] = class_id
    return mapped


def shapes_from_onnx_mask(
    *,
    mask: npt.NDArray[np.uint8],
    model_name: str,
    image_size: tuple[int, int],
    polygon_detail: int,
    keep_classes: Collection[str],
    minimum_polygon_area: int = 0,
) -> list[Shape]:
    if minimum_polygon_area < 0:
        raise ValueError("Minimum polygon area cannot be negative")
    detections = [
        Detection(
            mask=mask == class_id,
            label=label,
            description=f"ONNX: {model_name}",
        )
        for class_id, label in enumerate(CLASS_NAMES[1:], start=1)
        if label in keep_classes and np.any(mask == class_id)
    ]
    shapes = shapes_from_detections(
        detections=detections,
        shape_type="polygon",
        image_size=image_size,
        polygon_detail=polygon_detail,
    )
    shapes = [
        shape
        for shape in shapes
        if _polygon_area(shape.points) >= minimum_polygon_area
    ]
    # A semantic segmentation mask may contain disconnected islands. They are
    # independent labels in this workflow, so do not expose builder group IDs.
    for shape in shapes:
        shape.group_id = None
    return shapes


def _polygon_area(points: npt.NDArray[np.float64]) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


class OnnxSegmenter:
    def __init__(self) -> None:
        self._model_path: Path | None = None
        self._session: ort.InferenceSession | None = None

    def predict(
        self,
        *,
        image: npt.NDArray[np.uint8],
        model_path: Path,
        polygon_detail: int,
        keep_classes: Collection[str] = CLASS_NAMES[1:],
        minimum_polygon_area: int = 0,
    ) -> OnnxPrediction:
        keep_classes = set(keep_classes)
        unknown_classes = keep_classes.difference(CLASS_NAMES[1:])
        if unknown_classes:
            raise ValueError(
                f"Unknown ONNX classes to keep: {sorted(unknown_classes)}; "
                f"available classes are {list(CLASS_NAMES[1:])}"
            )
        model_path = model_path.resolve()
        if self._session is None or self._model_path != model_path:
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            self._model_path = model_path

        gray = np.asarray(Image.fromarray(image).convert("L"), dtype=np.uint8)
        gray = mask_oct_region(gray)
        input_meta = self._session.get_inputs()[0]
        tensor = prepare_onnx_input(gray, input_meta.shape)
        started = time.perf_counter()
        output = self._session.run(None, {input_meta.name: tensor})[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        mask = onnx_output_to_mask(output)
        mask = np.asarray(
            Image.fromarray(mask).resize(
                (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        mask = mask_oct_region(mask)
        shapes = shapes_from_onnx_mask(
            mask=mask,
            model_name=model_path.name,
            image_size=(image.shape[1], image.shape[0]),
            polygon_detail=polygon_detail,
            keep_classes=keep_classes,
            minimum_polygon_area=minimum_polygon_area,
        )
        return OnnxPrediction(
            shapes=shapes,
            provider=self._session.get_providers()[0],
            elapsed_ms=elapsed_ms,
            model_path=model_path,
        )
