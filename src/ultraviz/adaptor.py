"""Convert Ultralytics Results without importing torch or Ultralytics."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np

from .predictions import Array, Image, Instance, Prediction, validate_image
from .timing import inference_seconds


def as_numpy(value: Any) -> Array:
    """Detach tensors and transfer them to CPU before NumPy conversion."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _resize_map(data: Array, shape: tuple[int, int], *, letterbox: bool = False) -> Array:
    """Restore dense data; instance masks use centered Ultralytics letterboxing."""
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2 or min(data.shape) == 0:
        raise ValueError("Dense maps must have nonempty (H, W) dimensions")
    height, width = shape
    if data.shape == shape:
        return data
    if letterbox:
        gain = min(data.shape[0] / height, data.shape[1] / width)
        pad_x = (data.shape[1] - width * gain) / 2
        pad_y = (data.shape[0] - height * gain) / 2
        left, top = round(pad_x - 0.1), round(pad_y - 0.1)
        right, bottom = data.shape[1] - round(pad_x + 0.1), data.shape[0] - round(pad_y + 0.1)
        data = data[top:bottom, left:right]
    return np.asarray(
        cv2.resize(data.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    )


def from_ultralytics(
    result: Any,
    *,
    include_masks: bool = True,
    include_keypoints: bool = True,
) -> Prediction:
    """Adapt one Results object, including tracked boxes or tracked OBBs.

    Coordinates come from public Results properties. Instance masks preserve
    holes and remove standard centered letterbox padding before resizing.
    Disabled masks or keypoints are not transferred to CPU or converted.
    """
    image = getattr(result, "orig_img", None)
    validate_image(image)
    image = cast(Image, image)
    shape = image.shape[:2]
    raw_names = getattr(result, "names", {}) or {}
    names = raw_names if isinstance(raw_names, dict) else dict(enumerate(raw_names))
    obb = getattr(result, "obb", None)
    boxes = obb if obb is not None else getattr(result, "boxes", None)
    masks = getattr(result, "masks", None) if include_masks else None
    points = getattr(result, "keypoints", None) if include_keypoints else None
    instances = []
    if boxes is not None:
        raw = getattr(boxes, "data", None)
        if raw is not None:
            # Transfer metadata once instead of synchronizing each tensor slice.
            data = as_numpy(raw)
            expected = (7, 8) if obb is not None else (6, 7)
            if data.ndim != 2 or data.shape[1] not in expected:
                raise ValueError("Unexpected Ultralytics box data shape")
            classes, scores = data[:, -1], data[:, -2]
            tracked = data.shape[1] == expected[1]
            ids = data[:, -3] if tracked else None
            geometry = as_numpy(boxes.xyxyxyxy) if obb is not None else data[:, :4]
        else:
            # Results compatible adapters may expose only the public properties.
            classes = as_numpy(boxes.cls).reshape(-1)
            scores = as_numpy(boxes.conf).reshape(-1)
            ids = None if boxes.id is None else as_numpy(boxes.id).reshape(-1)
            geometry = as_numpy(boxes.xyxyxyxy if obb is not None else boxes.xyxy)
        mask_data = None if masks is None else as_numpy(masks.data)
        point_data = None if points is None else as_numpy(points.data)
        if not (len(classes) == len(scores) == len(geometry)):
            raise ValueError("Results geometry, classes, and confidence counts do not match")
        for values in (ids, mask_data, point_data):
            if values is not None and len(values) != len(classes):
                raise ValueError("Results instance counts do not match")
        for index, class_value in enumerate(classes):
            class_id = int(class_value)
            instances.append(
                Instance(
                    class_id=class_id,
                    confidence=float(scores[index]),
                    name=names.get(class_id, str(class_id)),
                    track_id=None if ids is None else int(ids[index]),
                    box=None if obb is not None else tuple(geometry[index].tolist()),
                    corners=geometry[index] if obb is not None else None,
                    mask=None
                    if mask_data is None
                    else _resize_map(mask_data[index], shape, letterbox=True) > 0.5,
                    keypoints=None if point_data is None else point_data[index],
                )
            )
    probs = getattr(result, "probs", None)
    classifications: tuple[tuple[str, float], ...] = ()
    if probs is not None:
        values = as_numpy(probs.data).reshape(-1)
        classifications = tuple(
            (names.get(int(i), str(i)), float(values[i])) for i in np.argsort(-values)[:5]
        )
    semantic = getattr(result, "semantic_mask", None) if include_masks else None
    depth = getattr(result, "depth", None)
    return Prediction(
        image=image,
        inference_seconds=inference_seconds(result),
        instances=tuple(instances),
        classifications=classifications,
        semantic=None if semantic is None else _resize_map(as_numpy(semantic.data), shape),
        depth=None if depth is None else _resize_map(as_numpy(depth.data), shape),
        source=str(result.path) if getattr(result, "path", None) is not None else None,
    )


def prediction_source(result: Any) -> str | None:
    """Read a result source path without transferring or copying prediction data."""
    source = result.source if isinstance(result, Prediction) else getattr(result, "path", None)
    return None if source is None else str(source)
