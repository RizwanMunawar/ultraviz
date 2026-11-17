# ruff: noqa: D103
"""Task summary cards use filtered live metrics and preserve frame boundaries."""

import numpy as np
import pytest

from ultraviz import Instance, Prediction, Visualizer, process


def test_count_updates_and_custom_callback_sees_readonly_context(arial):
    image = np.zeros((480, 800, 3), np.uint8)
    contexts = []

    def custom(state):
        contexts.append(state)
        with pytest.raises(TypeError):
            state.items["In frame"] = 99
        return {"Cars": state.class_counts.get("car", 0), "Status": "Running"}

    visualizer = Visualizer(
        demo="count",
        summary=True,
        summary_items=custom,
        min_confidence=0.5,
        labels=False,
        summary_width=300,
        summary_title="Vehicles summary",
    )
    result = Prediction(image, (Instance(0, name="car", track_id=4), Instance(1, confidence=0.1)))
    output = visualizer(result)
    assert visualizer.last_summary.items == {
        "In frame": 1,
        "Classes": 1,
        "Tracks": 1,
        "Cars": 1,
        "Status": "Running",
    }
    assert contexts[0].track_ids == (4,)
    assert not output[:, :484].any()
    assert not output[:16].any()
    assert not output[:, -16:].any()
    assert output[16:300, 484:784].any()
    assert not image.any()
    visualizer(Prediction(image))
    assert visualizer.last_summary.items["In frame"] == 0
    assert contexts[-1].frame == 2


@pytest.mark.parametrize(
    "task,expected",
    [
        ("detect", {"In frame": 1, "Classes": 1}),
        ("segment", {"Instances": 1, "Masks": 1, "Coverage": "25.0%"}),
        ("pose", {"Poses": 1, "Keypoints": 1}),
        ("obb", {"Oriented boxes": 1, "Classes": 1}),
        ("track", {"In frame": 1, "Classes": 1, "Tracks": 0}),
        ("heatmap", {"In frame": 1, "Samples": 1}),
        ("semantic", {"Classes": 1, "Coverage": "50.0%"}),
        ("classify", {"Top class": "car", "Confidence": "90.0%"}),
        ("depth", {"Valid pixels": "100.0%", "Minimum": "0", "Maximum": "1"}),
    ],
)
def test_task_metrics(task, expected, arial):
    image = np.zeros((240, 400, 3), np.uint8)
    mask = np.zeros((240, 400), bool)
    mask[:120, :200] = True
    instance = Instance(
        0,
        box=(20, 20, 80, 80),
        mask=mask if task == "segment" else None,
        keypoints=np.array([[30, 30, 0.9], [40, 40, 0.1]]) if task == "pose" else None,
        corners=np.array([[20, 20], [80, 20], [80, 80], [20, 80]]) if task == "obb" else None,
    )
    dense = task in ("semantic", "classify", "depth")
    semantic = np.zeros((240, 400), int)
    semantic[:120] = 255
    prediction = Prediction(
        image,
        () if dense else (instance,),
        semantic=semantic if task == "semantic" else None,
        depth=np.linspace(0, 1, 400)[None].repeat(240, axis=0) if task == "depth" else None,
        classifications=(("car", 0.9),) if task == "classify" else (),
    )
    visualizer = Visualizer(demo=task, labels=False)
    assert visualizer.style.summary is True
    visualizer(prediction)
    assert visualizer.last_summary.task == task
    assert visualizer.last_summary.items == expected


def test_static_metrics_are_copied_and_flat_process_options_work(arial):
    metrics = {"Status": "Ready"}
    visualizer = Visualizer(summary=True, summary_items=metrics, labels=False)
    metrics["Status"] = "Changed"
    output = process(
        Prediction(np.zeros((180, 320, 3), np.uint8)),
        summary=True,
        summary_items={"Status": "Working"},
        labels=False,
    )
    assert output.any()
    visualizer(Prediction(np.zeros((180, 320, 3), np.uint8)))
    assert visualizer.last_summary.items["Status"] == "Ready"


@pytest.mark.parametrize("shape", [(1, 1), (24, 40), (1080, 1920)])
def test_panel_fits_tiny_and_large_images(shape, arial):
    image = np.zeros((*shape, 3), np.uint8)
    output = Visualizer(summary=True, summary_items={"Long " * 10: "Value " * 100}, labels=False)(
        Prediction(image)
    )
    assert output.shape == image.shape


@pytest.mark.parametrize(
    "options",
    [
        {"summary": "yes"},
        {"summary_width": 0},
        {"summary_columns": 4},
        {"summary_opacity": 2},
        {"summary_title": ""},
        {"summary_margin": -1},
        {"summary_items": {"x": float("nan")}},
        {"summary_items": {"": 3}},
    ],
)
def test_invalid_summary_options_fail_early(options):
    with pytest.raises((ValueError, TypeError)):
        Visualizer(**options)


def test_callback_invalid_values_fail_when_frame_is_consumed(image):
    visualizer = Visualizer(summary=True, summary_items=lambda state: {"bad": None})
    with pytest.raises(ValueError, match="summary values"):
        visualizer(Prediction(image))


def test_disabled_summary_skips_metrics_and_fonts(image, monkeypatch):
    monkeypatch.setattr("ultraviz.annotation.frame_summary", lambda *a: pytest.fail("No summary"))
    visualizer = Visualizer(
        summary=False, labels=False, summary_items=lambda state: pytest.fail("No callback")
    )
    visualizer(Prediction(image))
    assert visualizer.last_summary is None


def test_default_depth_blends_and_zero_opacity_preserves_pixels():
    image = np.full((10, 10, 3), 80, np.uint8)
    image[5:] = 180
    depth = np.ones((10, 10))
    depth[0, 0] = np.nan
    output = Visualizer(summary=False)(Prediction(image, depth=depth))
    np.testing.assert_array_equal(output[0, 0], image[0, 0])
    assert (output[5] > output[4]).all()
    assert not np.array_equal(output[4], image[4])
    np.testing.assert_array_equal(
        Visualizer(summary=False, depth_opacity=0)(Prediction(image, depth=depth)), image
    )
    np.testing.assert_array_equal(image[:5], 80)


def test_raw_frame_summary_uses_dimensions_instead_of_fabricated_objects(image, arial):
    visualizer = Visualizer(labels=False)
    visualizer(image)
    assert visualizer.last_summary.task == "frame"
    assert visualizer.last_summary.items == {"Width": image.shape[1], "Height": image.shape[0]}


def test_summary_adapts_masks_even_with_mask_drawing_disabled(image, arial):
    from types import SimpleNamespace

    raw = SimpleNamespace(
        orig_img=image,
        names={0: "person"},
        boxes=SimpleNamespace(data=np.array([[10, 10, 60, 60, 0.9, 0]])),
        masks=SimpleNamespace(data=np.ones((1, *image.shape[:2]), dtype=np.float32)),
    )
    visualizer = Visualizer(summary=True, masks=False, labels=False)
    visualizer(raw)
    assert visualizer.last_summary.task == "segment"
    assert visualizer.last_summary.items["Coverage"] == "100.0%"


def test_classification_summary_hides_confidence_with_show_conf(image, arial):
    visualizer = Visualizer(summary=True, show_conf=False, labels=False)
    visualizer(Prediction(image, classifications=(("car", 0.9),)))
    assert visualizer.last_summary.items == {"Top class": "car"}


def test_heatmap_summary_accumulates_then_resets_with_sequence(image, arial):
    visualizer = Visualizer(demo="heatmap", summary=True, labels=False)
    result = Prediction(image, (Instance(0, box=(20, 20, 40, 40)),), source="one.mp4")
    visualizer(result)
    visualizer(result)
    assert visualizer.last_summary.items["Samples"] == 2
    visualizer(Prediction(image, source="two.mp4"))
    assert visualizer.last_summary.items["Samples"] == 0


def test_title_and_subtitle_are_direct_visualizer_keywords(arial, monkeypatch):
    from ultraviz.summary import SummaryPanel

    drawn = []
    original = SummaryPanel._text

    def capture(draw, position, text, font, rgb, width):
        drawn.append((text, position))
        return original(draw, position, text, font, rgb, width)

    monkeypatch.setattr(SummaryPanel, "_text", staticmethod(capture))
    visualizer = Visualizer(
        {"summary_title": "Configured title", "summary_subtitle": "Configured subtitle"},
        summary_title="Production overview",
        summary_subtitle="Ampoule counting",
        labels=False,
    )
    visualizer(Prediction(np.zeros((480, 800, 3), np.uint8)))
    assert drawn[0][0] == "Production overview"
    assert drawn[1][0] == "Ampoule counting"
    assert drawn[0][1][1] < drawn[1][1][1] < drawn[2][1][1]
    assert visualizer.style.summary_subtitle == "Ampoule counting"


@pytest.mark.parametrize("subtitle", ["", "  ", 42, False])
def test_invalid_subtitle_is_rejected(subtitle):
    with pytest.raises(ValueError, match="summary_subtitle"):
        Visualizer(summary_subtitle=subtitle)


@pytest.mark.parametrize("shape", [(1, 1), (24, 40), (1920, 1080), (1080, 1920)])
def test_subtitle_and_vertical_legend_fit_all_frame_sizes(shape, arial):
    image = np.zeros((*shape, 3), np.uint8)
    output = Visualizer(
        demo="heatmap", summary_title="Traffic overview", summary_subtitle="A long subtitle " * 10
    )(Prediction(image))
    assert output.shape == image.shape
    assert not image.any()


def test_heatmap_legend_has_high_values_above_low_values(arial):
    from ultraviz.branding import DARK_BLUE, NEON_PINK

    frame = Visualizer(demo="heatmap", heatmap_opacity=0)(
        Prediction(np.zeros((720, 1280, 3), np.uint8))
    )
    ys, xs = np.where(np.all(frame == NEON_PINK[::-1], axis=2))
    assert len(ys) > 0
    top = int(ys.min())
    column = int(np.median(xs))
    bottom = np.flatnonzero(np.all(frame[:, column] == DARK_BLUE[::-1], axis=1))
    assert bottom.max() - top > xs.max() - xs.min()
    assert bottom.max() > top
