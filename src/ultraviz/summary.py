"""Task metrics and a compact branded summary panel for annotated frames."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from .branding import DARK_BLUE, NEON_PINK, NEON_YELLOW, WHITE, text_color
from .predictions import Array, Image, Instance, Prediction

if TYPE_CHECKING:
    from .style import Style

Metric = str | int | float


@dataclass(frozen=True)
class FrameSummary:
    """Current filtered task metrics passed to summary_items callbacks.

    Track IDs describe this frame. Counts are visible objects, not cumulative
    unique objects or region crossings. No inference durations are included.
    """

    task: str
    frame: int
    items: Mapping[str, Metric]
    class_counts: Mapping[str, int]
    track_ids: tuple[int, ...]


def validate_items(items: Mapping[str, Metric]) -> dict[str, Metric]:
    """Copy finite scalar metrics and reject malformed callback or static values."""
    if not isinstance(items, Mapping):
        raise TypeError("summary_items must be a mapping or return a mapping")
    copied = dict(items)
    for label, value in copied.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("summary metric names must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("summary values must be strings, integers or finite floats")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("summary values must be finite")
    return copied


def frame_summary(
    prediction: Prediction,
    visible: list[Instance],
    frame: int,
    task: str | None,
    density: Array | None,
    style: Style,
) -> FrameSummary:
    """Compute meaningful task statistics only when a summary panel is requested."""
    if task is None:
        task = (
            "depth"
            if prediction.depth is not None
            else "semantic"
            if prediction.semantic is not None
            else "classify"
            if prediction.classifications
            else "pose"
            if any(i.keypoints is not None for i in visible)
            else "segment"
            if any(i.mask is not None for i in visible)
            else "obb"
            if any(i.corners is not None for i in visible)
            else "track"
            if any(i.track_id is not None for i in visible)
            else "detect"
        )
    counts = Counter(i.name if i.name is not None else str(i.class_id) for i in visible)
    ids = tuple(sorted({i.track_id for i in visible if i.track_id is not None}))
    items: dict[str, Metric] = {"In frame": len(visible), "Classes": len(counts)}
    if task == "frame":
        items = {"Width": prediction.image.shape[1], "Height": prediction.image.shape[0]}
    elif task == "heatmap":
        items = {
            "In frame": len(visible),
            "Samples": int(density.sum()) if density is not None else 0,
        }
    elif task == "segment":
        masks = [i.mask for i in visible if i.mask is not None]
        items = {"Instances": len(visible), "Masks": len(masks)}
        if masks:
            union = masks[0].copy()
            for mask in masks[1:]:
                union |= mask
            items["Coverage"] = f"{np.count_nonzero(union) / union.size:.1%}"
    elif task == "pose":
        points = [i.keypoints for i in visible if i.keypoints is not None]
        joints = 0
        for points_array in points:
            valid = np.any(points_array[:, :2] != 0, axis=1)
            if points_array.shape[1] == 3:
                valid &= points_array[:, 2] >= style.keypoint_confidence
            joints += int(np.count_nonzero(valid))
        items = {"Poses": len(points), "Keypoints": joints}
    elif task == "obb":
        items = {
            "Oriented boxes": sum(i.corners is not None for i in visible),
            "Classes": len(counts),
        }
    elif task == "semantic" and prediction.semantic is not None:
        semantic = prediction.semantic
        valid = ~np.isin(semantic, style.ignored_semantic_ids)
        if style.classes is not None:
            valid &= np.isin(semantic, tuple(style.classes))
        items = {"Classes": len(np.unique(semantic[valid])), "Coverage": f"{valid.mean():.1%}"}
    elif task == "depth" and prediction.depth is not None:
        finite = np.isfinite(prediction.depth)
        values = prediction.depth[finite]
        items = {"Valid pixels": f"{finite.mean():.1%}"}
        if values.size:
            items.update({"Minimum": f"{values.min():.3g}", "Maximum": f"{values.max():.3g}"})
    elif task == "classify":
        items = {
            "Top class": prediction.classifications[0][0] if prediction.classifications else "None"
        }
        if prediction.classifications and style.show_conf:
            items["Confidence"] = f"{prediction.classifications[0][1]:.1%}"
    if ids or task == "track":
        items["Tracks"] = len(ids)
    context = FrameSummary(
        task, frame, MappingProxyType(items.copy()), MappingProxyType(dict(counts)), ids
    )
    custom = style.summary_items(context) if callable(style.summary_items) else style.summary_items
    if custom is not None:
        items.update(validate_items(custom))
    return FrameSummary(task, frame, MappingProxyType(items), context.class_counts, ids)


class SummaryPanel:
    """Render a rounded top right panel without converting the complete frame to PIL."""

    def draw(
        self,
        image: Image,
        summary: FrameSummary,
        style: Style,
        font_for_size: Callable[[int], ImageFont.FreeTypeFont],
    ) -> None:
        """Fit a title, metric cards and track IDs within the available frame area."""
        height, width = image.shape[:2]
        margin = min(style.summary_margin, max(0, (min(height, width) - 1) // 4))
        available_width, available_height = width - 2 * margin, height - 2 * margin
        panel_width = min(available_width, style.summary_width or max(240, round(width * 0.28)))
        size = max(
            1, min(style.font_size or max(12, round(min(height, width) * 0.019)), panel_width // 16)
        )
        entries = list(summary.items.items())
        columns = min(style.summary_columns, max(1, len(entries)))
        rows = (len(entries) + columns - 1) // columns
        # Reduce the layout on small images rather than drawing outside the frame.
        while True:
            pad = max(2, round(size * 0.8))
            gap = max(2, size // 2)
            card_height = size * 4
            footer = size * 2 if summary.track_ids else 0
            panel_height = pad * 2 + size * 2 + rows * card_height + max(0, rows - 1) * gap + footer
            if panel_height <= available_height or size == 1:
                break
            size -= 1
        panel_height = min(available_height, panel_height)
        canvas = PILImage.new("RGBA", (panel_width, panel_height))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            (0, 0, panel_width - 1, panel_height - 1),
            radius=max(1, pad),
            fill=(*DARK_BLUE, round(255 * style.summary_opacity)),
        )
        title = style.summary_title or f"{summary.task.capitalize()} summary"
        self._text(draw, (pad, pad), title, font_for_size(size), WHITE, panel_width - 2 * pad)
        card_width = max(1, (panel_width - 2 * pad - (columns - 1) * gap) // columns)
        palette = (
            NEON_YELLOW,
            (118, 255, 214),
            (4, 42, 255),
            NEON_PINK,
            (0, 255, 255),
            (172, 249, 255),
        )
        for index, (label, value) in enumerate(entries):
            x = pad + (index % columns) * (card_width + gap)
            y = pad + size * 2 + (index // columns) * (card_height + gap)
            rgb = palette[index % len(palette)]
            draw.rounded_rectangle(
                (x, y, x + card_width - 1, y + card_height - 1),
                radius=max(1, gap),
                fill=(*rgb, 255),
            )
            inset = max(1, size // 2)
            text_width = max(1, card_width - 2 * inset)
            self._text(
                draw,
                (x + inset, y + inset),
                label,
                font_for_size(size),
                text_color(rgb),
                text_width,
            )
            self._text(
                draw,
                (x + inset, y + size * 2),
                str(value),
                font_for_size(size + max(1, size // 2)),
                text_color(rgb),
                text_width,
            )
        if summary.track_ids:
            text = "Track IDs: " + ", ".join(map(str, summary.track_ids))
            self._text(
                draw,
                (pad, panel_height - pad - size),
                text,
                font_for_size(size),
                WHITE,
                panel_width - 2 * pad,
            )
        rgba = np.asarray(canvas)
        x, y = width - margin - panel_width, margin
        roi = image[y : y + panel_height, x : x + panel_width]
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255
        roi[:] = np.rint(roi * (1 - alpha) + rgba[:, :, 2::-1] * alpha).astype(np.uint8)

    @staticmethod
    def _text(
        draw: ImageDraw.ImageDraw,
        position: tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        rgb: tuple[int, int, int],
        width: int,
    ) -> None:
        """Ellipsize long titles and values so text stays inside its allocated card."""
        if font.getlength(text) > width:
            low, high = 0, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if font.getlength(text[:middle] + "…") <= width:
                    low = middle
                else:
                    high = middle - 1
            text = text[:low] + "…" if font.getlength("…") <= width else ""
        draw.text(position, text, font=font, fill=(*rgb, 255), anchor="lt")
