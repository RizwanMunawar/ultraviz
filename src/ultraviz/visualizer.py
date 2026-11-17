"""Configure annotation or lazily infer a source with an explicit model."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .inference import predict, resolve_demo
from .predictions import Image
from .processing import ProcessSummary, _configure, _process
from .style import Style
from .summary import FrameSummary
from .timing import FrameTiming


class Visualizer(Iterator[Any]):
    """Visualize existing results, or expose a lazy source prediction stream.

    Visualizer(args)(results) renders existing predictions. For inference use
    Visualizer(source, model=model, ...). Iterating that instance yields raw
    Ultralytics Results. Call it with no arguments to render its source using
    configured show/save options, or pass it to process(..., show=True).

    Args accepts a dictionary, argparse Namespace or Style object. Keywords
    override those settings. Inference arguments belong in inference={...}.
    No model is discovered from caller variables or loaded implicitly.
    One instance owns one sequential stream and is not thread safe.
    """

    def __init__(
        self,
        args: str | Path | int | Mapping[str, Any] | Namespace | Style | None = None,
        *,
        source: str | Path | int | None = None,
        model: Any = None,
        inference: Mapping[str, Any] | None = None,
        tracking: bool | None = None,
        task: str | None = None,
        demo: str | None = None,
        on_result: Callable[[Any], None] | None = None,
        **options: Any,
    ) -> None:
        """Validate configuration without running inference or consuming frames."""
        if isinstance(args, (str, Path, int)) and not isinstance(args, bool):
            if source is not None:
                raise ValueError("Pass a source once, positionally or with source=")
            source, args = args, None
        if source is not None and (
            isinstance(source, bool) or not isinstance(source, (str, Path, int))
        ):
            raise TypeError("source must be an image/video path, URL or camera index")
        if source is not None and model is None:
            raise ValueError(
                "Source inference requires model=model; caller variables are not inspected"
            )
        if inference is not None and not isinstance(inference, Mapping):
            raise TypeError("inference must be a mapping of Ultralytics arguments")
        inference_options = dict(inference or {})
        nested_task = inference_options.pop("task", None)
        if task is not None and nested_task is not None and task != nested_task:
            raise ValueError("Conflicting task and inference['task']; select one task")
        if on_result is not None and not callable(on_result):
            raise TypeError("on_result must be callable")
        if isinstance(args, Style):
            settings: dict[str, Any] = {"style": args}
        elif isinstance(args, Namespace):
            settings = vars(args).copy()
        elif isinstance(args, Mapping):
            settings = dict(args)
        elif args is None:
            settings = {}
        else:
            raise TypeError("args must be a source, dictionary, Namespace or Style configuration")
        settings.update(options)
        configured_task = settings.pop("task", None)
        configured_tracking = settings.pop("tracking", False)
        selected_task = task if task is not None else nested_task
        configured_demo = settings.pop("demo", None)
        selected_demo = demo if demo is not None else configured_demo
        style_value = settings.get("style")
        if selected_demo is None and isinstance(style_value, Style):
            selected_demo = style_value.demo
        self.task, tracking = resolve_demo(
            selected_demo,
            selected_task if selected_task is not None else configured_task,
            configured_tracking if tracking is None else tracking,
        )
        if selected_demo is not None or self.task is not None or tracking:
            settings["demo"] = selected_demo or ("track" if tracking else self.task)
        self.show: bool = settings.pop("show", False)
        self.save: str | Path | None = settings.pop("save", None)
        self.fps: float = settings.pop("fps", 30)
        self.overwrite: bool = settings.pop("overwrite", False)
        style = settings.pop("style", None)
        self._annotator = _configure(
            style,
            settings,
            show=self.show,
            save=self.save,
            fps=self.fps,
            overwrite=self.overwrite,
        )
        self.source = source
        self.model = model
        self.tracking = tracking
        self.inference = inference_options
        reserved = {"source", "model", "tracking"}.intersection(self.inference)
        if reserved:
            raise ValueError(
                f"Pass {', '.join(sorted(reserved))} as Visualizer arguments, "
                "not inference settings"
            )
        if "stream" in self.inference:
            raise ValueError("Source inference always streams; do not pass inference['stream']")
        self.inference.setdefault("verbose", False)
        self.inference.setdefault("batch", 1)
        self.inference.setdefault("stream_buffer", False)
        self.on_result = on_result
        self.last_result: Any = None
        self._results: Iterator[Any] | None = None
        self._exhausted = False

    @property
    def style(self) -> Style:
        """Return immutable annotation settings, also inherited by process(self)."""
        return self._annotator.style

    @property
    def last_timing(self) -> FrameTiming | None:
        """Return the latest optional timing record without retaining frame history."""
        return self._annotator.last_timing

    @property
    def last_summary(self) -> FrameSummary | None:
        """Return the latest immutable task metrics when summary=True."""
        return self._annotator.last_summary

    @property
    def count(self) -> int:
        """Return the visible instance count from the most recently annotated frame."""
        return self._annotator.count

    def __iter__(self) -> Visualizer:
        """Return this single pass raw prediction stream without buffering frames."""
        if self.source is None:
            raise TypeError("Iteration requires Visualizer(source, model=model)")
        return self

    def __next__(self) -> Any:
        """Infer one frame lazily and expose the raw Results object to the caller."""
        if self._exhausted:
            raise StopIteration
        if self.source is None:
            raise TypeError("Iteration requires Visualizer(source, model=model)")
        try:
            if self._results is None:
                self._results = predict(
                    self.model,
                    self.source,
                    task=self.task,
                    tracking=self.tracking,
                    **self.inference,
                )
            self.last_result = next(self._results)
            if self.on_result is not None:
                self.on_result(self.last_result)
            return self.last_result
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release an inference generator on completion, early stopping or failure."""
        self._exhausted = True
        iterator, self._results = self._results, None
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    def __call__(
        self, results: Any = None
    ) -> Image | list[Image] | Iterator[Image] | ProcessSummary:
        """Render supplied predictions, or consume this instance's configured source.

        Show/save mode displays each frame immediately after inference and
        returns ProcessSummary. Without show/save, annotated frames stay lazy
        for a source or iterator. No completed video is replayed afterward.
        """
        if results is None:
            if self.source is None:
                raise TypeError("Pass results, or configure Visualizer(source, model=model)")
            results = self
        return _process(
            results,
            self._annotator,
            show=self.show,
            save=self.save,
            fps=self.fps,
            overwrite=self.overwrite,
        )

    def annotate(self, result: Any) -> Image:
        """Draw one result or copy one BGR frame, without saving or previewing."""
        return self._annotator.annotate(result)

    def stream(self, results: Iterable[Any], *, reset: bool = True) -> Iterator[Image]:
        """Yield annotated frames lazily and close upstream when stopping early."""
        return self._annotator.stream(results, reset=reset)

    def reset(self) -> None:
        """Clear annotation history without rewinding or restarting model inference."""
        self._annotator.reset()
