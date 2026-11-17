# ruff: noqa: D100, D103
# Scenario names document tests; library definitions require docstrings.
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ultraviz import Instance, Prediction, Style, Visualizer, from_ultralytics, process
from ultraviz.branding import PALETTE, color, text_color


def test_detection_does_not_mutate_input(image, style):
    item = Instance(0, 0.9, name="person", box=(20, 20, 70, 70))
    output = process(Prediction(image, (item,)), style=replace(style, line_width=2))
    assert tuple(output[20, 40]) == color(0)[::-1]
    assert not image.any()
    assert output is not image


def test_masks_preserve_holes(image, style):
    mask = np.zeros(image.shape[:2], bool)
    mask[10:80, 10:80] = True
    mask[30:60, 30:60] = False
    output = process(
        Prediction(image, (Instance(0, mask=mask),)), style=replace(style, mask_opacity=1)
    )
    assert tuple(output[15, 15]) == color(0)[::-1]
    assert not output[40, 40].any()
    assert not output[90, 90].any()


def test_obb_corners(image, style):
    corners = np.array([[40, 10], [80, 40], [50, 80], [10, 50]])
    output = process(Prediction(image, (Instance(3, corners=corners),)), style=style)
    assert output[25, 60].any()  # Rotated edge remains correctly positioned.
    assert not output[10, 40].any()  # The original sharp vertex is rounded away.
    assert not output[10, 10].any()


def test_pose_topology_confidence_and_unknown_layout(image, style):
    points = np.array([[20, 30, 0.9], [80, 30, 0.9], [80, 80, 0.1]])
    item = Instance(0, keypoints=points)
    output = process(Prediction(image, (item,)), style=replace(style, skeleton=((0, 1), (1, 2))))
    assert output[30, 50].any()
    assert not output[55, 80].any()
    output = process(Prediction(image, (item,)), style=style)
    assert output[30, 20].any()
    assert not output[30, 50].any()
    with pytest.raises(ValueError, match="skeleton index"):
        process(Prediction(image, (item,)), style=replace(style, skeleton=((0, 3),)))


def tracked(image, x, source="video.mp4"):
    return Prediction(image, (Instance(0, track_id=7, box=(x, 30, x + 10, 40)),), source=source)


def test_tracking_history_reset_and_expiry(image, style):
    visualizer = Visualizer(replace(style, boxes=False, trail_length=2, trail_max_gap=1))
    visualizer(tracked(image, 10))
    output = visualizer(tracked(image, 70))
    assert output[35, 40].any()
    assert len(visualizer._annotator._tracks[(0, 7)][1]) == 2
    output = visualizer(tracked(image, 90, source="other.mp4"))
    assert not output.any()
    visualizer(Prediction(image, source="other.mp4"))
    visualizer(Prediction(image, source="other.mp4"))
    assert not visualizer._annotator._tracks
    visualizer.reset()
    assert visualizer._annotator._frame == 0


def test_filters(image, style):
    items = (Instance(0, 0.8, box=(10, 10, 40, 40)), Instance(1, 0.1, box=(50, 50, 70, 70)))
    output = process(
        Prediction(image, items), style=replace(style, classes=frozenset({1}), min_confidence=0.5)
    )
    assert not output.any()


def test_stream_is_lazy_and_closes(image, style):
    events = []

    def results():
        try:
            for _ in range(3):
                events.append("next")
                yield Prediction(image)
        finally:
            events.append("closed")

    frames = Visualizer(style).stream(results())
    assert not events
    next(frames)
    assert events == ["next"]
    frames.close()
    assert events == ["next", "closed"]


def test_semantic_ignore_ids(image, style):
    semantic = np.full(image.shape[:2], 255)
    semantic[:20, :20] = 3
    output = process(Prediction(image, semantic=semantic), style=replace(style, mask_opacity=1))
    assert tuple(output[5, 5]) == color(3)[::-1]
    assert not output[50, 50].any()


def test_depth_invalid_and_constant(image, style):
    depth = np.ones(image.shape[:2])
    depth[0, 0] = np.nan
    depth[0, 1] = np.inf
    output = process(Prediction(image, depth=depth), style=style)
    assert not output[0, :2].any()
    assert output[30, 30].any()
    assert np.array_equal(output[30, 30], output[50, 50])


def test_empty_noncontiguous_image(image, style):
    view = image[:, ::-1]
    assert np.array_equal(process(Prediction(view), style=style), view)


def test_arial_label_and_unicode(image, arial):
    output = process(Prediction(image, (Instance(0, name="café", box=(20, 40, 100, 70)),)))
    assert ((output == np.array(color(0)[::-1])).all(axis=2)).any()


def test_palette_contrast():
    assert len(PALETTE) == 20
    assert color(0) == (4, 42, 255)
    assert color(20) == color(0)
    assert text_color((4, 42, 255)) == (255, 255, 255)
    assert text_color((243, 243, 243)) == (17, 31, 104)


@pytest.mark.parametrize(
    "option,value",
    [
        ("mask_opacity", float("nan")),
        ("line_width", True),
        ("font_size", 0),
        ("min_confidence", 2),
        ("trail_length", -1),
        ("skeleton", ((-1, 2),)),
    ],
)
def test_style_validation(option, value):
    with pytest.raises(ValueError):
        Style(**{option: value})


@pytest.mark.parametrize(
    "options",
    [
        {"class_id": -1},
        {"class_id": 0, "confidence": float("inf")},
        {"class_id": 0, "box": (30, 20, 10, 40)},
        {"class_id": 0, "corners": [[1, 2]]},
        {"class_id": 0, "keypoints": [[float("nan"), 2]]},
    ],
)
def test_instance_validation(options):
    with pytest.raises(ValueError):
        Instance(**options)


def test_prediction_validation(image):
    with pytest.raises(ValueError, match="uint8"):
        Prediction(image.astype(float))
    with pytest.raises(ValueError, match="dimensions"):
        Prediction(image, (Instance(0, mask=np.ones((2, 2))),))
    with pytest.raises(ValueError, match="integer"):
        Prediction(image, semantic=np.full(image.shape[:2], 0.5))


def test_buffers_copied():
    corners = np.zeros((4, 2))
    item = Instance(0, corners=corners)
    corners[:] = 10
    assert not item.corners.any()
    assert not item.corners.flags.writeable


def test_letterboxed_masks(image):
    mask = np.zeros((1, 128, 128))
    mask[0, 16:112, :64] = 1
    boxes = NS(
        cls=np.array([2]), conf=np.array([0.8]), id=np.array([42]), xyxy=np.array([[0, 0, 64, 96]])
    )
    result = NS(orig_img=image, boxes=boxes, masks=NS(data=mask), names={2: "car"})
    item = from_ultralytics(result).instances[0]
    assert item.track_id == 42
    assert item.mask[:, :64].all()
    assert not item.mask[:, 64:].any()
    assert item.name == "car"


def test_missing_arial_error(tmp_path):
    from ultraviz.style import load_font

    with pytest.raises(FileNotFoundError, match="Arial is required"):
        load_font(Style(font_path=tmp_path / "missing.ttf"))


def test_dense_single_row_not_squeezed():
    image = np.zeros((1, 8, 3), dtype=np.uint8)
    semantic = np.zeros((1, 8), dtype=np.int64)
    result = NS(orig_img=image, names={}, semantic_mask=NS(data=semantic))
    assert from_ultralytics(result).semantic.shape == (1, 8)


def test_bad_adapter_image():
    with pytest.raises(ValueError, match="uint8 BGR"):
        from_ultralytics(NS(orig_img=None))


def test_call_accepts_prediction_list_and_tuple(image, style):
    renderer = Visualizer(style)
    item = Prediction(image, (Instance(0, box=(20, 20, 70, 70)),))
    for batch in ([item], (item,), [item, item], []):
        frames = renderer(batch)
        assert isinstance(frames, list)
        assert len(frames) == len(batch)
        for frame in frames:
            assert frame.shape == image.shape
            assert frame.any()


def test_call_accepts_stream_lazily_and_closes(image, style):
    events = []

    def results():
        try:
            events.append("read")
            yield Prediction(image)
        finally:
            events.append("closed")

    frames = Visualizer(style)(results())
    assert not events
    assert next(frames).shape == image.shape
    frames.close()
    assert events == ["read", "closed"]


def test_configuration_requires_style():
    with pytest.raises(TypeError, match="Style configuration"):
        Visualizer(object())


def test_font_family_is_verified(monkeypatch, tmp_path):
    from ultraviz.style import load_font

    font_path = tmp_path / "Arial.ttf"
    font_path.write_bytes(b"substitute font")
    monkeypatch.setattr(
        "ultraviz.style.ImageFont.truetype",
        lambda *args: NS(getname=lambda: ("Archivo", "Regular")),
    )
    with pytest.raises(ValueError, match="require Arial"):
        load_font(Style(font_path=font_path))


def test_batch_calls_reset_trails(image, style):
    visualizer = Visualizer(replace(style, boxes=False))
    visualizer([tracked(image, 10)])
    assert not visualizer([tracked(image, 70)])[0].any()
    # Incremental single-result calls still retain trails.
    assert visualizer(tracked(image, 90)).any()


def test_rounded_box_corners_and_square_override(image, style):
    item = Prediction(image, (Instance(0, box=(20, 20, 80, 80)),))
    rounded = process(item, style=replace(style, line_width=1, corner_radius=12))
    square = process(item, style=replace(style, line_width=1, corner_radius=0))
    assert not rounded[20, 20].any()
    assert square[20, 20].any()
    assert rounded[20, 50].any()
    assert rounded[50, 20].any()


def test_auto_font_size_and_rounded_labels(arial):
    small = np.zeros((360, 640, 3), np.uint8)
    large = np.zeros((1080, 1920, 3), np.uint8)
    visualizer = Visualizer(boxes=False)
    item = Instance(0, name="person", box=(80, 80, 200, 200))
    small_output = visualizer(Prediction(small, (item,)))
    small_size = visualizer._annotator._font.size
    large_output = visualizer(Prediction(large, (item,)))
    large_size = visualizer._annotator._font.size
    assert small_size == 10
    assert large_size == 27
    assert not small_output[80, 80].any()  # Rounded bottom left of the label fill.
    assert small_output.any() and large_output.any()
    explicit = Visualizer(font_size=16)
    explicit(Prediction(large, (item,)))
    assert explicit._annotator._font.size == 16


@pytest.mark.parametrize("box", [(0, 0, 1, 1), (-10, -10, 20, 20), (40, 20, 40, 80)])
def test_tiny_clipped_and_degenerate_rounded_boxes(image, style, box):
    output = process(Prediction(image, (Instance(0, box=box),)), style=style)
    assert output.shape == image.shape
    assert output.dtype == np.uint8
