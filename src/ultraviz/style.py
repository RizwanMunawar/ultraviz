"""Validated rendering settings and strict Arial font discovery."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from PIL import ImageFont

from .branding import RGB
from .inference import DEMOS
from .summary import FrameSummary, Metric, validate_items

# Zero-based COCO 17-point topology. Other layouts need explicit edges.
COCO_SKELETON = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


@dataclass(frozen=True)
class Style:
    """Rendering options with Ultralytics class colors and optional RGB text color.

    font_path must point to a licensed Arial TrueType font. No substitute font
    is silently used. Disable labels if Arial is unavailable.
    """

    boxes: bool = True
    labels: bool = True
    show_conf: bool = True
    text_label_color: RGB | None = None
    timing: bool | str = False
    demo: str | None = None
    summary: bool = False
    summary_title: str | None = None
    summary_items: Mapping[str, Metric] | Callable[[FrameSummary], Mapping[str, Metric]] | None = (
        None
    )
    summary_width: int | None = None
    summary_columns: int = 2
    summary_margin: int = 16
    summary_opacity: float = 0.92
    depth_opacity: float = 0.45
    track_ids: bool = True
    masks: bool = True
    pose: bool = True
    trails: bool = True
    line_width: int | None = None
    font_size: int | None = None
    corner_radius: int = 8
    label_radius: int = 6
    label_padding: int = 6
    label_margin: int = 0
    font_path: str | Path | None = None
    mask_opacity: float = 0.45
    keypoint_radius: int | None = None
    keypoint_confidence: float = 0.25
    skeleton: tuple[tuple[int, int], ...] | None = None
    trail_length: int = 10
    trail_max_gap: int = 10
    classes: frozenset[int] | None = None
    min_confidence: float = 0.0
    ignored_semantic_ids: tuple[int, ...] = (255, -1)

    def __post_init__(self) -> None:
        """Validate annotation sizes, thresholds, class filters and skeleton indices."""
        if self.timing is True:
            object.__setattr__(self, "timing", "console")
        elif self.timing is not False and self.timing not in ("console",):
            raise ValueError("timing must be False, True or console; timings are console logs only")
        if self.summary_title is not None and (
            not isinstance(self.summary_title, str) or not self.summary_title.strip()
        ):
            raise ValueError("summary_title must be a nonempty string or None")
        if self.summary_items is not None and not callable(self.summary_items):
            object.__setattr__(
                self, "summary_items", MappingProxyType(validate_items(self.summary_items))
            )
        if self.summary_width is not None and (
            isinstance(self.summary_width, bool)
            or not isinstance(self.summary_width, int)
            or self.summary_width < 1
        ):
            raise ValueError("summary_width must be a positive integer or None")
        if (
            isinstance(self.summary_columns, bool)
            or not isinstance(self.summary_columns, int)
            or not 1 <= self.summary_columns <= 3
        ):
            raise ValueError("summary_columns must be between 1 and 3")
        if self.text_label_color is not None:
            rgb = self.text_label_color
            if (
                not isinstance(rgb, (tuple, list, np.ndarray))
                or len(rgb) != 3
                or any(
                    isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v <= 255
                    for v in rgb
                )
            ):
                raise ValueError(
                    "text_label_color must be an RGB tuple of three integers from 0 to 255"
                )
            object.__setattr__(self, "text_label_color", tuple(int(v) for v in rgb))
        for name in (
            "boxes",
            "labels",
            "show_conf",
            "track_ids",
            "masks",
            "pose",
            "trails",
            "summary",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("trail_length", "trail_max_gap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.demo is not None and self.demo not in DEMOS:
            raise ValueError(f"demo must be one of {', '.join(DEMOS)}")
        if self.trail_length > 10:
            raise ValueError("trail_length must be between 1 and 10 frames")
        if self.line_width is not None and (
            isinstance(self.line_width, bool)
            or not isinstance(self.line_width, int)
            or self.line_width < 1
        ):
            raise ValueError("line_width must be a positive integer or None for automatic sizing")
        if self.keypoint_radius is not None and (
            isinstance(self.keypoint_radius, bool)
            or not isinstance(self.keypoint_radius, int)
            or self.keypoint_radius < 1
        ):
            raise ValueError(
                "keypoint_radius must be a positive integer or None for automatic sizing"
            )
        if self.font_size is not None and (
            isinstance(self.font_size, bool)
            or not isinstance(self.font_size, int)
            or self.font_size < 1
        ):
            raise ValueError("font_size must be a positive integer or None for automatic sizing")
        for name in (
            "corner_radius",
            "label_radius",
            "label_padding",
            "label_margin",
            "summary_margin",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "mask_opacity",
            "depth_opacity",
            "summary_opacity",
            "keypoint_confidence",
            "min_confidence",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if self.classes is not None:
            if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.classes):
                raise ValueError("classes must contain nonnegative integer class IDs")
            object.__setattr__(self, "classes", frozenset(self.classes))
        if self.skeleton is not None:
            edges = tuple(tuple(edge) for edge in self.skeleton)
            if any(
                len(edge) != 2
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in edge)
                for edge in edges
            ):
                raise ValueError("skeleton edges must contain two nonnegative integer indices")
            object.__setattr__(self, "skeleton", edges)

    def resolved_font_size(
        self,
        shape: tuple[int, int],
        box_size: tuple[float, float] | None = None,
    ) -> int:
        """Scale 18 px Arial at a 720 px short edge, retaining readable small labels.

        Portrait and landscape frames use the same short edge scale. Large
        images keep scaling beyond 4K; explicit font_size remains a fixed size.
        """
        if self.font_size is not None:
            return self.font_size
        size = max(10, round(18 * min(shape) / 720))
        if box_size is not None:
            limit = max(10, round(min(box_size[0] * 0.3, box_size[1] * 0.5)))
            if limit < size:
                size = max(10, limit // 2 * 2)
        return size

    def resolved_line_width(
        self,
        shape: tuple[int, int],
        box_size: tuple[float, float] | None = None,
    ) -> int:
        """Scale a 2 px outline at a 720 px short edge and thin small object outlines.

        Use the short edge so rotating a frame does not change its styling.
        Explicit line_width remains a fixed pixel width at every resolution.
        """
        if self.line_width is not None:
            return self.line_width
        width = max(1, round(2 * min(shape) / 720))
        if box_size is not None:
            width = min(width, max(1, round(min(box_size) * 0.06)))
        return width


def load_font(style: Style, *, size: int | None = None) -> ImageFont.FreeTypeFont:
    """Resolve Arial on macOS, Windows, Linux, or from ULTRAVIZ_FONT."""
    explicit = style.font_path or os.environ.get("ULTRAVIZ_FONT")
    candidates = (
        [Path(explicit)]
        if explicit
        else [
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf",
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"),
            Path.home() / ".fonts/Arial.ttf",
        ]
    )
    for path in candidates:
        if path.is_file():
            font = ImageFont.truetype(str(path), size or style.font_size or 18)
            if (font.getname()[0] or "").lower() != "arial":
                raise ValueError(f"Brand labels require Arial; {path} is {font.getname()[0]}")
            return font
    raise FileNotFoundError(
        "Arial is required by the Ultralytics annotation brand guide. Set ULTRAVIZ_FONT "
        "or Style(font_path=...) to your licensed Arial.ttf, or use Style(labels=False)."
    )
