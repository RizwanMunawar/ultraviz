# ruff: noqa: D103
"""Demo selection, adaptive styling, confidence toggles and short trail windows."""

from types import SimpleNamespace

import numpy as np
import pytest

from ultraviz import Instance, Prediction, Style, Visualizer
from ultraviz.annotation import Annotator
from ultraviz.branding import NEON_PINK, NEON_YELLOW
from ultraviz.inference import TASKS


@pytest.mark.parametrize("demo", TASKS)
def test_demo_selects_matching_inference_mode(demo):
    calls = []
    model_task = "detect" if demo == "track" else demo
    model = SimpleNamespace(
        task=model_task,
        predict=lambda **options: calls.append("predict") or iter(["result"]),
        track=lambda **options: calls.append("track") or iter(["result"]),
    )
    source = Visualizer("input.mp4", model=model, demo=demo)
    assert next(source) == "result"
    assert calls == ["track" if demo == "track" else "predict"]
    assert source.model is model
    source.close()


def test_unknown_demo_and_conflicting_task_fail_early():
    with pytest.raises(ValueError, match="demo must be one of"):
        Visualizer(demo="unknown")
    with pytest.raises(ValueError, match="demo and task"):
        Visualizer(task="pose", demo="segment")


def test_inline_model_construction_is_reused_without_reloading(monkeypatch):
    import ultraviz.visualizer as api

    model = SimpleNamespace(task="detect")
    seen = []
    monkeypatch.setattr(
        api,
        "predict",
        lambda passed, *args, **kwargs: seen.append(passed) or iter(["first", "second"]),
    )
    source = Visualizer("input.mp4", model=model, demo="detect")
    assert list(source) == ["first", "second"]
    assert seen == [model]


def test_show_conf_only_changes_text(image, monkeypatch):
    captured = []
    monkeypatch.setattr(
        Annotator,
        "_labels",
        lambda self, frame, labels: captured.extend(text for text, *_ in labels) or frame,
    )
    prediction = Prediction(image, (Instance(0, name="person", confidence=0.87),))
    visualizer = Visualizer(show_conf=False)
    visualizer(prediction)
    assert captured == ["person"]
    assert visualizer.count == 1
    assert prediction.instances[0].confidence == 0.87
    assert Visualizer(confidence=False).style.show_conf is False
    with pytest.raises(ValueError, match="legacy alias"):
        Visualizer(confidence=False, show_conf=True)


def test_default_label_touches_box_while_retaining_padding(arial):
    image = np.zeros((360, 640, 3), np.uint8)
    prediction = Prediction(image, (Instance(0, name="person", box=(100, 150, 300, 300)),))
    visualizer = Visualizer(show_conf=False)
    output = visualizer(prediction)
    assert visualizer.style.label_margin == 0
    assert visualizer.style.label_padding == 6
    assert output[148:152, 120].any(axis=1).all()


def test_default_font_and_line_width_consider_image_and_box_sizes():
    style = Style()
    assert style.resolved_line_width((1080, 1920)) == 4
    assert style.resolved_line_width((2160, 3840)) == 8
    assert style.resolved_line_width((2160, 3840), (20, 12)) == 1
    assert style.resolved_font_size((1080, 1920)) == 30
    assert Style(line_width=4, font_size=24).resolved_line_width((96, 128), (20, 12)) == 4
    assert Style(font_size=24).resolved_font_size((96, 128)) == 24


def test_trails_keep_at_most_ten_actual_frames(image):
    visualizer = Visualizer(labels=False, boxes=False)
    for index in range(30):
        visualizer(Prediction(image, (Instance(0, track_id=7, box=(index, 20, index + 5, 30)),)))
    points = visualizer._annotator._tracks[(0, 7)][1]
    assert len(points) == 10
    assert points[-1][0] - points[0][0] == 9
    with pytest.raises(ValueError, match="between 1 and 10"):
        Style(trail_length=11)
    # Missing detections must not stretch a trail across more than ten frames.
    for _ in range(9):
        visualizer(Prediction(image))
    visualizer(Prediction(image, (Instance(0, track_id=7, box=(80, 20, 90, 30)),)))
    assert len(visualizer._annotator._tracks[(0, 7)][1]) == 1


def test_depth_uses_exact_yellow_to_pink_endpoints(image):
    depth = np.linspace(0, 1, image.shape[1])[None, :].repeat(image.shape[0], axis=0)
    output = Visualizer(demo="depth", depth_opacity=1)(Prediction(image, depth=depth))
    assert tuple(output[0, 0]) == NEON_YELLOW[::-1]
    assert tuple(output[0, -1]) == NEON_PINK[::-1]


def test_count_is_current_visible_objects_and_uses_shared_labels(image, monkeypatch):
    captured = []
    monkeypatch.setattr(
        Annotator,
        "_labels",
        lambda self, frame, labels: captured.extend(text for text, *_ in labels) or frame,
    )
    visualizer = Visualizer(demo="count", min_confidence=0.5, show_conf=False)
    visualizer(Prediction(image, (Instance(0, confidence=0.9), Instance(1, confidence=0.2))))
    assert visualizer.count == 1
    assert "Count: 1" in captured
    visualizer(Prediction(image))
    assert visualizer.count == 0
    assert captured[-1] == "Count: 0"


def test_heatmap_has_bounded_spatial_history_and_resets(image):
    visualizer = Visualizer(demo="heatmap", boxes=False, labels=False)
    prediction = Prediction(image, (Instance(0, box=(30, 30, 50, 50)),))
    first = visualizer(prediction)
    for _ in range(20):
        output = visualizer(prediction)
    density = visualizer._annotator._density
    assert density.shape == (12, 16)
    assert density.sum() == 21
    assert first.any() and output.any()
    visualizer.reset()
    assert visualizer._annotator._density is None


@pytest.mark.parametrize("demo", ["count", "heatmap"])
def test_instance_demos_reject_dense_model_outputs(image, demo):
    with pytest.raises(ValueError, match="instance predictions"):
        Visualizer(demo=demo)(Prediction(image, semantic=np.zeros(image.shape[:2], int)))


def test_rotated_label_attaches_to_actual_rounded_outline(arial):
    image = np.zeros((400, 400, 3), np.uint8)
    corners = np.array([[200.0, 150.0], [250.0, 200.0], [200.0, 250.0], [150.0, 200.0]])
    output = Visualizer(show_conf=False)(
        Prediction(image, (Instance(0, name="car", corners=corners),))
    )
    assert output[151:155, 200].any(axis=1).all()
