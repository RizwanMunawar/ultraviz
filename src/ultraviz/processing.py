"""Standard single call API for rendering, previewing and exporting predictions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast, overload

from .adaptor import prediction_source
from .annotation import Annotator
from .export import _protect_source, _validate_fps, save_image, save_video
from .predictions import Image
from .preview import PreviewWindow
from .style import Style

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".avi"}


@dataclass(frozen=True)
class ProcessSummary:
    """Completed display/export operation; output is a file, directory or None."""

    frames: int
    output: Path | None
    stopped: bool = False


@overload
def process(
    results: list[Any] | tuple[Any, ...],
    *,
    show: Literal[False] = False,
    save: None = None,
    fps: float = 30,
    overwrite: bool = False,
    style: Style | None = None,
    **style_options: Any,
) -> list[Image]: ...


@overload
def process(
    results: Iterator[Any],
    *,
    show: Literal[False] = False,
    save: None = None,
    fps: float = 30,
    overwrite: bool = False,
    style: Style | None = None,
    **style_options: Any,
) -> Iterator[Image]: ...


@overload
def process(
    results: Any,
    *,
    show: Literal[False] = False,
    save: None = None,
    fps: float = 30,
    overwrite: bool = False,
    style: Style | None = None,
    **style_options: Any,
) -> Image: ...


@overload
def process(
    results: Any,
    *,
    show: bool = False,
    save: str | Path | None = None,
    fps: float = 30,
    overwrite: bool = False,
    style: Style | None = None,
    **style_options: Any,
) -> Image | list[Image] | Iterator[Image] | ProcessSummary: ...


def process(
    results: Any,
    *,
    show: bool = False,
    save: str | Path | None = None,
    fps: float = 30,
    overwrite: bool = False,
    style: Style | None = None,
    **style_options: Any,
) -> Image | list[Image] | Iterator[Image] | ProcessSummary:
    """Render, display or save image/video results through one entry point.

    Style fields can be passed directly, e.g. line_width=3, font_size=20,
    masks=True, boxes=False. They override fields in an optional Style object.
    Unknown options raise TypeError. Class colors and Arial remain enforced.

    With no show/save, return an image, image list or lazy frame iterator,
    matching the input. With show/save, consume results immediately and return
    ProcessSummary without retaining frames. MP4/AVI paths save video; image
    paths require one result; directory paths save numbered PNG frames.
    FPS defaults to 30 for playback and export; no audio is exported.
    Press Q/Escape to stop previewing; any key closes the final frame.
    Each call starts a fresh tracking sequence. Use Visualizer for incremental
    calls. Close lazy output when stopping consumption early.
    """
    # Import locally because the public class shares this processing implementation.
    from .visualizer import Visualizer

    if isinstance(results, Visualizer) and style is None and not style_options:
        _validate_output(show=show, save=save, fps=fps, overwrite=overwrite)
        renderer = results._annotator
    else:
        if isinstance(results, Visualizer) and style is None:
            style = results.style
        renderer = _configure(
            style, style_options, show=show, save=save, fps=fps, overwrite=overwrite
        )
    return _process(results, renderer, show=show, save=save, fps=fps, overwrite=overwrite)


def _configure(
    style: Style | None,
    style_options: dict[str, Any],
    *,
    show: bool,
    save: str | Path | None,
    fps: float,
    overwrite: bool,
) -> Annotator:
    """Validate configuration and construct one reusable annotation engine."""
    _validate_output(show=show, save=save, fps=fps, overwrite=overwrite)
    if style is not None and not isinstance(style, Style):
        raise TypeError("style must be a Style configuration")
    if "confidence" in style_options:
        if "show_conf" in style_options:
            raise ValueError("Use show_conf once; confidence is its legacy alias")
        style_options["show_conf"] = style_options.pop("confidence")
    on_timing = style_options.pop("on_timing", None)
    return Annotator(
        replace(style, **style_options) if style else Style(**style_options), on_timing=on_timing
    )


def _validate_output(*, show: bool, save: str | Path | None, fps: float, overwrite: bool) -> None:
    """Validate output settings without constructing a duplicate annotation engine."""
    if not isinstance(show, bool) or not isinstance(overwrite, bool):
        raise TypeError("show and overwrite must be booleans")
    if save is not None and not isinstance(save, (str, Path)):
        raise TypeError("save must be a file/directory path, e.g. save='annotated.mp4'")
    _validate_fps(fps)
    target = Path(save) if save is not None else None
    suffix = target.suffix.lower() if target is not None else ""
    if target is not None and suffix and suffix not in _IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS:
        raise ValueError(
            "save must be an image path, .mp4/.avi path, or directory without a suffix"
        )


def _process(
    results: Any,
    renderer: Annotator,
    *,
    show: bool,
    save: str | Path | None,
    fps: float,
    overwrite: bool,
) -> Image | list[Image] | Iterator[Image] | ProcessSummary:
    """Consume predictions with the configured engine and requested output behavior."""
    if not show and save is None:
        return cast(Image | list[Image] | Iterator[Image], renderer(results))

    target = Path(save) if save is not None else None
    suffix = target.suffix.lower() if target is not None else ""
    iterator = iter(results) if isinstance(results, (list, tuple, Iterator)) else iter((results,))
    preview = PreviewWindow(show, fps)
    try:
        if target is not None and suffix in _VIDEO_EXTENSIONS:
            target.parent.mkdir(parents=True, exist_ok=True)
            video = save_video(
                iterator,
                target,
                fps=fps,
                visualizer=renderer,
                overwrite=overwrite,
                on_frame=preview,
            )
            preview.finish()
            return ProcessSummary(video.frames, target, preview.stopped)

        if isinstance(results, (list, tuple, Iterator)):
            renderer.reset()
        if target is not None and suffix in _IMAGE_EXTENSIONS:
            first = next(iterator, None)
            if first is None:
                raise ValueError("Cannot save an empty image stream")
            if next(iterator, None) is not None:
                raise ValueError("Multiple results require a directory or .mp4/.avi save path")
            iterator_to_render = iter((first,))
        else:
            iterator_to_render = iterator
        count = 0
        for result in iterator_to_render:
            output = None
            if target is not None:
                output = target if suffix else target / f"frame_{count + 1:06d}.png"
                if count == 0:
                    output.parent.mkdir(parents=True, exist_ok=True)
                _protect_source(output, prediction_source(result))
            frame = renderer.annotate(result)
            if output is not None:
                save_image(frame, output, overwrite=overwrite)
            count += 1
            if not preview(frame):
                break
        preview.finish()
        return ProcessSummary(count, target, preview.stopped)
    finally:
        try:
            preview.close()
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
