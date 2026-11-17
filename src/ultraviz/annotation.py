"""Task composition and bounded, per-sequence tracking state."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator
from time import perf_counter
from typing import Any, cast, overload

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from .adaptor import from_ultralytics
from .branding import DARK_BLUE, NEON_PINK, NEON_YELLOW, RGB, color, text_color
from .geometry import rounded_polygon, rounded_rectangle
from .predictions import Array, Image, Instance, Prediction, validate_image
from .style import COCO_SKELETON, Style, load_font
from .summary import FrameSummary, SummaryPanel, frame_summary
from .timing import FrameTiming, inference_seconds

Label = tuple[str, tuple[int, int], RGB, int, bool]


class Annotator:
    """Reusable renderer. One instance represents one sequential video stream.

    Single results return BGR arrays, lists/tuples return lists of arrays,
    and iterators return lazy frame iterators. Rendering never saves, displays,
    or runs inference. Call reset() between unrelated sequences
    that do not provide distinct source paths. Not thread safe.
    """

    def __init__(
        self,
        style: Style | None = None,
        *,
        on_timing: Callable[[FrameTiming], None] | None = None,
    ) -> None:
        """Resolve annotation style, initialize tracking and defer Arial loading until needed."""
        if style is not None and not isinstance(style, Style):
            raise TypeError("Visualizer expects a Style configuration, e.g. Visualizer(Style())")
        if on_timing is not None and not callable(on_timing):
            raise TypeError("on_timing must be callable")
        self.on_timing = on_timing
        self.style = style or Style()
        self._font: ImageFont.FreeTypeFont | None = None
        self._fonts: OrderedDict[int, ImageFont.FreeTypeFont] = OrderedDict()
        self._label_cache: OrderedDict[tuple[str, RGB, int, int, int], tuple[Image, Array]] = (
            OrderedDict()
        )
        self._label_cache_bytes = 0
        self._timing_frame = 0
        self.last_timing: FrameTiming | None = None
        self.reset()

    def reset(self) -> None:
        """Discard trail history; call when seeking or changing a video source."""
        self._frame = 0
        self._sequence: tuple[str | None, tuple[int, ...]] | None = None
        self._tracks: dict[tuple[int, int], tuple[int, deque[tuple[int, int, int]]]] = {}
        self._density: Array | None = None
        self.count = 0
        self.last_summary: FrameSummary | None = None

    @overload
    def __call__(self, result: list[Any] | tuple[Any, ...]) -> list[Image]: ...

    @overload
    def __call__(self, result: Iterator[Any]) -> Iterator[Image]: ...

    @overload
    def __call__(self, result: Any) -> Image: ...

    def __call__(self, result: Any) -> Image | list[Image] | Iterator[Image]:
        """Render a single result, a prediction batch, or a lazy video stream.

        Lists/tuples always return lists, even for zero or one result. A new
        batch resets tracking state; use annotate() for incremental frames.
        Iterators stay lazy and must be consumed or explicitly closed.
        """
        if isinstance(result, (list, tuple)):
            self.reset()
            return [self.annotate(item) for item in result]
        if isinstance(result, Iterator):
            return self.stream(result)
        return self.annotate(result)

    def stream(self, results: Iterable[Any], *, reset: bool = True) -> Iterator[Image]:
        """Yield one frame at a time and close the upstream iterator on exit.

        Close this generator explicitly when stopping early, or use
        contextlib.closing. Each new stream resets tracking by default.
        """
        if reset:
            self.reset()
        iterator = iter(results)
        try:
            for result in iterator:
                yield self.annotate(result)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def annotate(self, result: Prediction | Any) -> Image:
        """Render one frame and optionally report inference and visualization time."""
        if not self.style.timing and self.on_timing is None:
            return self._annotate_frame(result)
        start = perf_counter()
        image = self._annotate_frame(result)
        elapsed = perf_counter() - start
        self._timing_frame += 1
        timing = FrameTiming(self._timing_frame, inference_seconds(result), elapsed)
        self.last_timing = timing
        if self.style.timing:
            print(timing.text(), flush=True)
        if self.on_timing is not None:
            self.on_timing(timing)
        return image

    def _annotate_frame(self, result: Prediction | Any) -> Image:
        """Compose masks, boxes/OBBs, pose, trails, labels and dense maps."""
        raw_frame = isinstance(result, np.ndarray)
        if isinstance(result, np.ndarray):
            validate_image(result)
            self.count = 0
            if self.style.demo in ("count", "heatmap"):
                raise ValueError("count and heatmap need prediction results, not raw frames")
            if not self.style.summary:
                return cast(Image, result.copy())
            result = Prediction(result)
        prediction: Prediction = (
            result
            if isinstance(result, Prediction)
            else from_ultralytics(
                result,
                include_masks=self.style.summary
                or (self.style.masks and self.style.mask_opacity > 0),
                include_keypoints=self.style.pose or self.style.summary,
            )
        )
        sequence = (prediction.source, prediction.image.shape)
        if self._sequence != sequence:
            self.reset()
            self._sequence = sequence
        self._frame += 1
        if self._tracks:
            self._tracks = {
                key: value
                for key, value in self._tracks.items()
                if self._frame - value[0] <= self.style.trail_max_gap
            }
        if (
            self.style.demo in ("count", "heatmap")
            and (
                prediction.classifications
                or prediction.semantic is not None
                or prediction.depth is not None
            )
            and not prediction.instances
        ):
            raise ValueError(
                "count and heatmap require instance predictions from detect, segment, pose or obb"
            )
        image = cast(Image, prediction.image.copy())
        if prediction.depth is not None:
            self._depth(image, prediction.depth)
        if prediction.semantic is not None and self.style.masks:
            for class_id in np.unique(prediction.semantic):
                cls = int(class_id)
                if cls in self.style.ignored_semantic_ids or not self._visible(cls, 1.0):
                    continue
                self._mask(image, prediction.semantic == cls, color(cls))
        visible = [
            item for item in prediction.instances if self._visible(item.class_id, item.confidence)
        ]
        self.count = len(visible)
        if self.style.demo == "heatmap":
            self._heatmap(image, visible)
        if self.style.masks:
            for item in visible:
                if item.mask is not None:
                    self._mask(image, item.mask, color(item.class_id))
        labels: list[Label] = []
        for item in visible:
            rgb = color(item.class_id)
            bgr = rgb[::-1]
            anchor = (8, 8)
            box_size = None
            if item.box is not None:
                box_size = (item.box[2] - item.box[0], item.box[3] - item.box[1])
            elif item.corners is not None:
                edges = np.linalg.norm(np.roll(item.corners, -1, axis=0) - item.corners, axis=1)
                box_size = (float(edges[0]), float(edges[1]))
            line_width = self.style.resolved_line_width(image.shape[:2], box_size)
            if item.corners is not None:
                points = self._points(item.corners)
                path = rounded_polygon(points, self.style.corner_radius)
                top = float(np.min(path[:, 1]))
                top_x = float(np.mean(path[np.abs(path[:, 1] - top) <= 0.5, 0]))
                anchor = tuple(self._points(np.array([top_x, top])).tolist())
                if self.style.boxes:
                    self._outline(image, path, bgr, line_width)
            elif item.box is not None:
                points = self._points(np.asarray(item.box).reshape(2, 2))
                anchor = tuple(points[0].tolist())
                if self.style.boxes:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    path = rounded_rectangle(int(x2 - x1), int(y2 - y1), self.style.corner_radius)
                    self._outline(image, path + points[0], bgr, line_width)
            if self.style.pose and item.keypoints is not None:
                self._pose(image, item.keypoints, rgb, line_width)
            if self.style.trails and item.track_id is not None:
                self._trail(image, item, bgr, line_width)
            if self.style.labels:
                if item.track_id is not None:
                    if not self.style.track_ids:
                        continue
                    text = str(item.track_id)
                else:
                    text = item.name if item.name is not None else str(item.class_id)
                    if self.style.show_conf:
                        text += f" {item.confidence:.2f}"
                labels.append(
                    (
                        text,
                        anchor,
                        rgb,
                        self.style.resolved_font_size(image.shape[:2]),
                        item.corners is not None,
                    )
                )
        if self.style.labels:
            pitch = (
                self.style.resolved_font_size(image.shape[:2])
                + 2 * self.style.label_padding
                + self.style.label_margin
            )
            for rank, (name, confidence) in enumerate(prediction.classifications):
                text = f"{name} {confidence:.2f}" if self.style.show_conf else name
                labels.append(
                    (
                        text,
                        (self.style.label_margin, self.style.label_margin + (rank + 1) * pitch),
                        DARK_BLUE,
                        self.style.resolved_font_size(image.shape[:2]),
                        False,
                    )
                )
        if self.style.demo == "count" and self.style.labels and not self.style.summary:
            size = self.style.resolved_font_size(image.shape[:2])
            labels.append(
                (
                    f"Count: {self.count}",
                    (8, size + 2 * self.style.label_padding + 8),
                    DARK_BLUE,
                    size,
                    False,
                )
            )
        if labels:
            image = self._labels(image, labels)
        if self.style.summary:
            task = "frame" if raw_frame else self.style.demo
            if task is None and not isinstance(result, Prediction):
                for attribute, name in (
                    ("obb", "obb"),
                    ("keypoints", "pose"),
                    ("masks", "segment"),
                    ("probs", "classify"),
                ):
                    if getattr(result, attribute, None) is not None:
                        task = name
                        break
            self.last_summary = frame_summary(
                prediction, visible, self._frame, task, self._density, self.style
            )
            SummaryPanel().draw(image, self.last_summary, self.style, self._font_for_size)
        return image

    def _visible(self, class_id: int, confidence: float) -> bool:
        """Check whether class and confidence filters allow the instance."""
        return confidence >= self.style.min_confidence and (
            self.style.classes is None or class_id in self.style.classes
        )

    @staticmethod
    def _points(points: Array) -> Array:
        # Bound offscreen coordinates before converting to OpenCV int32.
        """Round pixel coordinates and bound them before OpenCV integer conversion."""
        return np.asarray(np.rint(np.clip(points, -1_000_000, 1_000_000)), dtype=np.int32)

    def _outline(self, image: Image, path: Array, bgr: RGB, line_width: int) -> None:
        """Draw an outline using fixed point coordinates and antialiasing."""
        fixed = np.asarray(np.rint(np.clip(path, -1_000_000, 1_000_000) * 16), dtype=np.int32)
        cv2.polylines(image, [fixed], True, bgr, line_width, cv2.LINE_AA, shift=4)

    def _mask(self, image: Image, mask: Array, rgb: RGB) -> None:
        """Blend a class color into selected mask pixels without changing the background."""
        opacity = self.style.mask_opacity
        if opacity == 0:
            return
        x, y, width, height = cv2.boundingRect(mask.view(np.uint8))
        if width == 0 or height == 0:
            return
        region = image[y : y + height, x : x + width]
        selected = mask[y : y + height, x : x + width]
        if opacity == 1:
            region[selected] = rgb[::-1]
        else:
            region[selected] = np.rint(
                region[selected] * (1 - opacity) + np.array(rgb[::-1]) * opacity
            )

    def _pose(self, image: Image, keypoints: Array, rgb: RGB, line_width: int) -> None:
        """Draw visible keypoints and supported skeleton edges in the class color."""
        visible = np.any(keypoints[:, :2] != 0, axis=1)
        if keypoints.shape[1] == 3:
            visible &= keypoints[:, 2] >= self.style.keypoint_confidence
        points = self._points(keypoints[:, :2])
        edges = self.style.skeleton
        if edges is None:
            edges = COCO_SKELETON if len(points) == 17 else ()
        for start, end in edges:
            if max(start, end) >= len(points):
                raise ValueError("skeleton index exceeds the number of keypoints")
            if visible[start] and visible[end]:
                cv2.line(
                    image,
                    tuple(points[start]),
                    tuple(points[end]),
                    rgb[::-1],
                    line_width,
                    cv2.LINE_AA,
                )
        for point in points[visible]:
            cv2.circle(
                image,
                tuple(point),
                self.style.keypoint_radius or max(2, line_width + 1),
                rgb[::-1],
                -1,
                cv2.LINE_AA,
            )

    def _trail(self, image: Image, instance: Instance, bgr: RGB, line_width: int) -> None:
        """Append a tracked center and draw its bounded history."""
        if instance.corners is not None:
            center = instance.corners.mean(axis=0)
        elif instance.box is not None:
            center = np.array(instance.box).reshape(2, 2).mean(axis=0)
        else:
            return
        key = (instance.class_id, int(instance.track_id))  # type: ignore[arg-type]
        _, points = self._tracks.get(key, (self._frame, deque(maxlen=self.style.trail_length)))
        while points and self._frame - points[0][0] >= self.style.trail_length:
            points.popleft()
        x, y = self._points(center).tolist()
        points.append((self._frame, x, y))
        self._tracks[key] = (self._frame, points)
        if len(points) > 1:
            cv2.polylines(
                image,
                [np.array([(x, y) for _, x, y in points], dtype=np.int32)],
                False,
                bgr,
                line_width,
                cv2.LINE_AA,
            )

    def _depth(self, image: Image, depth: Array) -> None:
        """Overlay finite depth values using a gradient from Ultralytics brand colors."""
        if self.style.depth_opacity == 0:
            return
        finite = np.isfinite(depth)
        if not finite.any():
            return
        values = depth[finite].astype(np.float64)
        lo, hi = float(np.min(values, axis=None)), float(np.max(values, axis=None))
        if hi == lo:
            scaled = np.zeros_like(values)
        elif np.isfinite(hi - lo):
            scaled = (values - lo) / (hi - lo)
        else:
            magnitude = max(abs(lo), abs(hi))
            scaled = (values / magnitude - lo / magnitude) / (hi / magnitude - lo / magnitude)
        rgb = np.array(NEON_YELLOW) + scaled[:, None] * (
            np.array(NEON_PINK) - np.array(NEON_YELLOW)
        )
        opacity = self.style.depth_opacity
        image[finite] = np.rint(image[finite] * (1 - opacity) + rgb[:, ::-1] * opacity)

    def _heatmap(self, image: Image, instances: list[Instance]) -> None:
        """Accumulate object centers in a bounded coarse grid and overlay brand colors."""
        height, width = image.shape[:2]
        if self._density is None:
            self._density = np.zeros(((height + 7) // 8, (width + 7) // 8), dtype=np.int64)
        centers = []
        for instance in instances:
            if instance.box is not None:
                x1, y1, x2, y2 = instance.box
                centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
            elif instance.corners is not None:
                centers.append(tuple(instance.corners.mean(axis=0).tolist()))
        for x, y in centers:
            if 0 <= x < width and 0 <= y < height:
                self._density[int(y) // 8, int(x) // 8] += 1
        if self.style.mask_opacity == 0 or not self._density.any():
            return
        smooth = cv2.GaussianBlur(self._density.astype(np.float32), (0, 0), 2.0)
        intensity = smooth / max(float(np.max(smooth)), 1e-12)
        rgb = np.asarray(color(0), dtype=np.float32) + intensity[:, :, None] * (
            np.asarray(NEON_PINK, dtype=np.float32) - np.asarray(color(0), dtype=np.float32)
        )
        tint = cv2.resize(np.rint(rgb[:, :, ::-1]).astype(np.uint8), (width, height))
        alpha = (
            cv2.resize(np.sqrt(intensity), (width, height))[:, :, None] * self.style.mask_opacity
        )
        blended = image.astype(np.float32)
        blended *= 1 - alpha
        blended += tint * alpha
        image[:] = np.rint(blended)

    def _font_for_size(self, size: int) -> ImageFont.FreeTypeFont:
        """Reuse a bounded set of Arial sizes for mixed object dimensions."""
        if size not in self._fonts:
            self._fonts[size] = load_font(self.style, size=size)
            if len(self._fonts) > 32:
                self._fonts.popitem(last=False)
        self._fonts.move_to_end(size)
        self._font = self._fonts[size]
        return self._font

    def _labels(self, image: Image, labels: list[Label]) -> Image:
        """Composite cached rounded Arial labels into their local image regions."""
        height, width = image.shape[:2]
        padding, margin = self.style.label_padding, self.style.label_margin
        inset_x, inset_y = min(margin, (width - 1) // 2), min(margin, (height - 1) // 2)
        available_width, available_height = width - 2 * inset_x, height - 2 * inset_y
        for text, (x, y), rgb, size, centered in labels:
            key = (text, rgb, size, width, height)
            cached = self._label_cache.get(key)
            if cached is None:
                font = self._font_for_size(size)
                bounds = font.getbbox(text)
                label_width = max(
                    1, min(available_width, int(np.ceil(bounds[2] - bounds[0])) + 2 * padding)
                )
                label_height = max(
                    1, min(available_height, int(np.ceil(bounds[3] - bounds[1])) + 2 * padding)
                )
                canvas = PILImage.new("RGBA", (label_width, label_height))
                draw = ImageDraw.Draw(canvas)
                draw.rounded_rectangle(
                    (0, 0, label_width - 1, label_height - 1),
                    radius=min(self.style.label_radius, label_width // 2, label_height // 2),
                    fill=(*rgb, 255),
                )
                draw.text(
                    (padding - bounds[0], padding - bounds[1]),
                    text,
                    font=font,
                    fill=(*(self.style.text_label_color or text_color(rgb)), 255),
                )
                rgba = np.asarray(canvas)
                bgr = cast(Image, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
                mask = np.array(rgba[:, :, 3], copy=True)
                cached = (bgr, mask)
                sprite_bytes = bgr.nbytes + mask.nbytes
                if sprite_bytes <= 8 * 1024 * 1024:
                    self._label_cache[key] = cached
                    self._label_cache_bytes += sprite_bytes
                    while len(self._label_cache) > 256 or self._label_cache_bytes > 8 * 1024 * 1024:
                        _, (old_bgr, old_mask) = self._label_cache.popitem(last=False)
                        self._label_cache_bytes -= old_bgr.nbytes + old_mask.nbytes
            if key in self._label_cache:
                self._label_cache.move_to_end(key)
            bgr, mask = cached
            label_height, label_width = bgr.shape[:2]
            if centered:
                x -= label_width // 2
            x = max(inset_x, min(x, width - inset_x - label_width))
            above = y - label_height - margin
            preferred_y = above if above >= inset_y else y + margin
            y = max(inset_y, min(preferred_y, height - inset_y - label_height))
            cv2.copyTo(bgr, mask, image[y : y + label_height, x : x + label_width])
        return image
