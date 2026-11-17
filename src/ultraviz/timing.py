"""Per frame inference metadata and measured visualization durations in seconds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

from .predictions import Prediction


@dataclass(frozen=True)
class FrameTiming:
    """Core frame durations, excluding preview, encoding and timing instrumentation.

    Missing inference metadata remains None. Total is available only when
    inference is known and equals inference plus visualization time.
    """

    frame: int
    inference_seconds: float | None
    visualization_seconds: float

    @property
    def total_seconds(self) -> float | None:
        """Return the sum of known inference and measured visualization durations."""
        if self.inference_seconds is None:
            return None
        return self.inference_seconds + self.visualization_seconds

    def text(self) -> str:
        """Format a concise console summary with explicit seconds and missing values."""
        return (
            f"Frame {self.frame} | inference: {format_seconds(self.inference_seconds)} | "
            f"visualization: {format_seconds(self.visualization_seconds)} | "
            f"total: {format_seconds(self.total_seconds)}"
        )


def format_seconds(value: float | None) -> str:
    """Format a duration without presenting unknown inference time as zero."""
    return "n/a" if value is None else f"{value:.6f} s"


def inference_seconds(result: Any) -> float | None:
    """Read Ultralytics milliseconds or explicitly supplied native seconds."""
    if isinstance(result, Prediction):
        return None if result.inference_seconds is None else float(result.inference_seconds)
    speed = getattr(result, "speed", None)
    if not isinstance(speed, Mapping):
        return None
    value = speed.get("inference")
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        return None
    return float(value) / 1000
