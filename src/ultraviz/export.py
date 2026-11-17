"""Explicit file output with validation, atomic publication and cleanup."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .adaptor import prediction_source
from .annotation import Annotator
from .predictions import Image, validate_image


class SupportsAnnotation(Protocol):
    """Rendering contract accepted by video export, without coupling to its UI."""

    def reset(self) -> None:
        """Clear the renderer's sequence history."""
        ...

    def annotate(self, result: Any) -> Image:
        """Return an annotated BGR image for one prediction."""
        ...


def _validate_fps(fps: float) -> None:
    """Require a finite positive real frame rate and reject boolean flags."""
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float, np.integer, np.floating))
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("fps must be finite and positive")


def _protect_source(path: Path, source: str | None) -> None:
    """Reject an output path that resolves to a known input source."""
    if source is not None and Path(source).resolve() == path.resolve():
        raise ValueError("output must not overwrite the input source")


def _check_target(path: Path, overwrite: bool) -> None:
    """Reject directories, existing targets and missing output directories."""
    if path.is_dir():
        raise IsADirectoryError(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")


def _publish(temporary: Path, target: Path, overwrite: bool) -> None:
    """Publish a completed temporary file while respecting overwrite policy."""
    if overwrite:
        os.replace(temporary, target)
    else:
        # Hard link atomically refuses an existing target, including races.
        os.link(temporary, target)


def save_image(image: Image, output: str | Path, *, overwrite: bool = False) -> Path:
    """Save a BGR image. Existing files are protected unless overwrite=True."""
    validate_image(image)
    target = Path(output)
    _check_target(target, overwrite)
    if target.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        raise ValueError("Unsupported image extension")
    success, encoded = cv2.imencode(target.suffix, image)
    if not success:
        raise OSError(f"Could not encode image: {target}")
    handle, name = tempfile.mkstemp(prefix=".ultraviz-", suffix=target.suffix, dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded.tobytes())
        _publish(temporary, target, overwrite)
    finally:
        temporary.unlink(missing_ok=True)
    return target


@dataclass(frozen=True)
class VideoSummary:
    """Metadata for a completed video export; frames are not retained."""

    path: Path
    frames: int
    fps: float
    size: tuple[int, int]


def save_video(
    results: Iterable[Any],
    output: str | Path,
    *,
    fps: float,
    visualizer: SupportsAnnotation | None = None,
    overwrite: bool = False,
    on_frame: Callable[[Image], bool] | None = None,
) -> VideoSummary:
    """Render a stream to MP4/AVI without retaining frames or opening a GUI.

    FPS must be supplied explicitly. Videos have no audio. Source paths from
    Predictions/Results are protected even when overwrite=True. The target is
    published only after success; errors preserve existing output files.
    Optional on_frame runs after each write; return False to stop exporting.
    """
    _validate_fps(fps)
    target = Path(output)
    _check_target(target, overwrite)
    codecs = {".mp4": "mp4v", ".avi": "MJPG"}
    if target.suffix.lower() not in codecs:
        raise ValueError("Video output must use .mp4 or .avi")
    renderer = visualizer or Annotator()
    renderer.reset()
    iterator = iter(results)
    writer = None
    temporary = None
    count = 0
    size = (0, 0)
    last_source: str | None = None
    resolved_target = target.resolve()
    try:
        for result in iterator:
            source = prediction_source(result)
            if source is not None and source != last_source:
                _protect_source(resolved_target, source)
                last_source = source
            frame = renderer.annotate(result)
            frame_size = (frame.shape[1], frame.shape[0])
            if writer is None:
                size = frame_size
                if any(value % 2 for value in size):
                    raise ValueError("Video dimensions must be even to avoid codec cropping")
                handle, name = tempfile.mkstemp(
                    prefix=".ultraviz-", suffix=target.suffix, dir=target.parent
                )
                os.close(handle)
                temporary = Path(name)
                writer = cv2.VideoWriter(
                    str(temporary),
                    cv2.VideoWriter_fourcc(*codecs[target.suffix.lower()]),
                    fps,
                    size,
                )
                if not writer.isOpened():
                    raise OSError("Video codec is unavailable")
            if frame_size != size:
                raise ValueError("All video frames must have the same dimensions")
            writer.write(frame)
            count += 1
            if on_frame is not None and not on_frame(frame):
                break
        if writer is None or temporary is None:
            raise ValueError("Cannot save an empty video stream")
        writer.release()
        writer = None
        if temporary.stat().st_size == 0:
            raise OSError("Video writer produced an empty file")
        _publish(temporary, target, overwrite)
    finally:
        try:
            if writer is not None:
                writer.release()
        finally:
            try:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
    return VideoSummary(target, count, fps, size)
