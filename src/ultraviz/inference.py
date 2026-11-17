"""Optional Ultralytics inference with explicit task compatibility checks."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

TASKS = ("detect", "segment", "pose", "obb", "classify", "semantic", "depth", "track")
DEMOS = TASKS + ("count", "heatmap")
_TRACKABLE = frozenset(("detect", "segment", "pose", "obb"))


def resolve_task(task: str | None, tracking: bool) -> tuple[str | None, bool]:
    """Validate task selection; track is a mode applied to a compatible model."""
    if not isinstance(tracking, bool):
        raise TypeError("tracking must be a boolean")
    if task is not None and task not in TASKS:
        raise ValueError(f"task must be one of {', '.join(TASKS)}, or None for automatic selection")
    if task == "track":
        return None, True
    if tracking and task is not None and task not in _TRACKABLE:
        raise ValueError("Tracking requires detect, segment, pose or obb weights")
    return task, tracking


def resolve_demo(demo: str | None, task: str | None, tracking: bool) -> tuple[str | None, bool]:
    """Map a requested demo to model task selection or an annotation mode."""
    if demo is not None and demo not in DEMOS:
        raise ValueError(f"demo must be one of {', '.join(DEMOS)}, or None for automatic rendering")
    task, tracking = resolve_task(task, tracking)
    if demo == "track":
        tracking = True
    elif demo in TASKS:
        if task is not None and task != demo:
            raise ValueError("demo and task must select compatible model weights")
        task = demo
    if demo in ("count", "heatmap") and task in ("classify", "semantic", "depth"):
        raise ValueError(
            "count and heatmap require instance predictions: detect, segment, pose or obb"
        )
    return resolve_task(task, tracking)


def predict(
    model: Any,
    source: Any,
    *,
    task: str | None = None,
    tracking: bool = False,
    **options: Any,
) -> Iterator[Any]:
    """Return streamed predictions using matching model weights and task.

    Omitting task uses the model's task. Explicit tasks validate weights rather
    than converting them. Tracking can be selected with task='track', or combined
    with a specific model task using tracking=True. Other options pass directly
    to Ultralytics; no model weights or precision settings are changed.
    """
    task, tracking = resolve_task(task, tracking)
    if "stream" in options:
        raise ValueError("predict() always uses stream=True; do not pass stream")
    if isinstance(model, (str, Path)):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Install inference support with: pip install 'ultraviz[ultralytics]'"
            ) from exc
        model = YOLO(str(model), task=task)
    actual_task = getattr(model, "task", None)
    if task is not None and actual_task is not None and actual_task != task:
        raise ValueError(
            f"Selected task '{task}' requires matching weights; model task is '{actual_task}'"
        )
    if tracking and actual_task is not None and actual_task not in _TRACKABLE:
        raise ValueError(
            f"Tracking is unsupported for '{actual_task}'; use detect, segment, pose or obb"
        )
    method = model.track if tracking else model.predict
    return iter(method(source=source, stream=True, **options))
