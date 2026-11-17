"""Small, validated data objects independent of a model framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]
Array = npt.NDArray[Any]


def validate_image(image: object) -> None:
    """Require a nonempty HWC uint8 BGR image."""
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) == 0
    ):
        raise ValueError("image must be a nonempty uint8 BGR array of shape (H, W, 3)")


@dataclass(frozen=True)
class Instance:
    """One object's image space geometry; masks must match the image size.

    Keypoints have shape (K, 2) or (K, 3), with optional confidence.
    OBB corners have shape (4, 2). Class IDs and track IDs are nonnegative.
    Arrays are copied on construction to decouple them from model buffers.
    """

    class_id: int
    confidence: float = 1.0
    name: str | None = None
    track_id: int | None = None
    box: tuple[float, float, float, float] | None = None
    corners: Array | None = None
    mask: Array | None = None
    keypoints: Array | None = None

    def __post_init__(self) -> None:
        """Validate geometry and copy arrays to isolate prediction data."""
        for field in ("class_id", "track_id"):
            value = getattr(self, field)
            if (field == "class_id" or value is not None) and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.name is not None and not isinstance(self.name, str):
            raise ValueError("name must be a string or None")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and between 0 and 1")
        if self.box is not None:
            box = tuple(float(x) for x in self.box)
            if len(box) != 4 or not np.isfinite(box).all() or box[2] < box[0] or box[3] < box[1]:
                raise ValueError("box must be finite ordered (x1, y1, x2, y2)")
            object.__setattr__(self, "box", box)
        for field in ("corners", "mask", "keypoints"):
            raw = getattr(self, field)
            if raw is None:
                continue
            array = np.asarray(raw)
            if array.dtype.kind not in "biuf":
                raise ValueError(f"{field} must contain real numeric values")
            if array.dtype != np.bool_ and not np.isfinite(array).all():
                raise ValueError(f"{field} must contain finite values")
            if field == "corners" and array.shape != (4, 2):
                raise ValueError("corners must have shape (4, 2)")
            if field == "mask":
                if array.ndim != 2:
                    raise ValueError("mask must have shape (H, W)")
            array = np.array(array, dtype=bool if field == "mask" else None, copy=True)
            if field == "keypoints" and (array.ndim != 2 or array.shape[1] not in (2, 3)):
                raise ValueError("keypoints must have shape (K, 2) or (K, 3)")
            array.flags.writeable = False
            object.__setattr__(self, field, array)


@dataclass(frozen=True)
class Prediction:
    """A frame and its instances, classification scores, or dense predictions.

    The image is borrowed and never modified by the renderer. Dense maps must
    have the image's (H, W) shape. NaN/Inf depth pixels are left unchanged.
    """

    image: Image
    instances: tuple[Instance, ...] = ()
    classifications: tuple[tuple[str, float], ...] = ()
    semantic: Array | None = None
    depth: Array | None = None
    source: str | None = None
    inference_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate geometry and copy arrays to isolate prediction data."""
        validate_image(self.image)
        if self.inference_seconds is not None and (
            isinstance(self.inference_seconds, bool)
            or not isinstance(self.inference_seconds, (int, float, np.integer, np.floating))
            or not np.isfinite(self.inference_seconds)
            or self.inference_seconds < 0
        ):
            raise ValueError("inference_seconds must be finite and nonnegative, or None")
        object.__setattr__(self, "instances", tuple(self.instances))
        for instance in self.instances:
            if instance.mask is not None and instance.mask.shape != self.image.shape[:2]:
                raise ValueError("instance masks must match the image dimensions")
        for field in ("semantic", "depth"):
            value = getattr(self, field)
            if value is not None:
                array = np.array(value, copy=True)
                if array.dtype.kind not in "biuf":
                    raise ValueError(f"{field} must contain real numeric values")
                if array.shape != self.image.shape[:2]:
                    raise ValueError(f"{field} must match the image dimensions")
                if (
                    field == "semantic"
                    and array.dtype.kind not in "biu"
                    and (not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all())
                ):
                    raise ValueError("semantic must contain finite integer class IDs")
                array.flags.writeable = False
                object.__setattr__(self, field, array)
        object.__setattr__(
            self, "classifications", tuple(tuple(item) for item in self.classifications)
        )
        for name, score in self.classifications:
            if not isinstance(name, str):
                raise ValueError("classification names must be strings")
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("classification scores must be finite and between 0 and 1")
