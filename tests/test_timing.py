# ruff: noqa: D103
"""Per frame timing uses explicit seconds and flat visualizer settings."""

from types import SimpleNamespace

import numpy as np
import pytest

from ultraviz import Instance, Prediction, Visualizer, from_ultralytics, process


def result(image, value=31.0):
    return SimpleNamespace(orig_img=image, names={}, path="video.mp4", speed={"inference": value})


def test_console_timing_converts_ms_and_measures_each_frame(image, monkeypatch, capsys):
    clock = iter([1.0, 1.006, 2.0, 2.004])
    monkeypatch.setattr("ultraviz.annotation.perf_counter", lambda: next(clock))
    visualizer = Visualizer(timing=True, labels=False, summary=False)
    visualizer(result(image))
    assert visualizer.last_timing.inference_seconds == 0.031
    assert visualizer.last_timing.visualization_seconds == pytest.approx(0.006)
    assert visualizer.last_timing.total_seconds == pytest.approx(0.037)
    visualizer(result(image))
    assert visualizer.last_timing.frame == 2
    assert visualizer.last_timing.total_seconds == pytest.approx(0.035)
    output = capsys.readouterr().out
    assert (
        "Frame 1 | inference: 0.031000 s | visualization: 0.006000 s | total: 0.037000 s" in output
    )
    assert "Frame 2" in output


def test_disabled_timing_does_not_read_clock(image, monkeypatch, capsys):
    def reject():
        raise AssertionError("Disabled timing should not read the clock")

    monkeypatch.setattr("ultraviz.annotation.perf_counter", reject)
    visualizer = Visualizer(labels=False, summary=False)
    visualizer(result(image))
    assert visualizer.last_timing is None
    assert not capsys.readouterr().out


@pytest.mark.parametrize("value", [None, -1.0, float("nan"), float("inf"), True, "bad"])
def test_unknown_inference_is_not_reported_as_zero(image, value, capsys):
    visualizer = Visualizer(labels=False, timing=True, summary=False)
    visualizer(result(image, value))
    assert visualizer.last_timing.inference_seconds is None
    assert visualizer.last_timing.total_seconds is None
    assert "inference: n/a" in capsys.readouterr().out


def test_callback_interleaves_with_frames_and_does_not_store_history(image, capsys):
    events = []

    def results():
        for frame in range(2):
            events.append(f"read {frame + 1}")
            yield Prediction(image, inference_seconds=0.02)

    visualizer = Visualizer(
        labels=False,
        on_timing=lambda record: events.append(f"timing {record.frame}"),
        summary=False,
    )
    frames = visualizer(results())
    assert not events
    list(frames)
    assert events == ["read 1", "timing 1", "read 2", "timing 2"]
    assert visualizer.last_timing.frame == 2
    assert not capsys.readouterr().out


def test_console_timings_never_change_frame_pixels_or_load_fonts(image, monkeypatch, capsys):
    monkeypatch.setattr(
        "ultraviz.annotation.load_font", lambda *a, **k: pytest.fail("No timing font")
    )
    visualizer = Visualizer(labels=False, timing=True, summary=False)
    output = visualizer(Prediction(image, inference_seconds=0.03))
    np.testing.assert_array_equal(output, image)
    assert "visualization:" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["overlay", "both"])
def test_frame_timing_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="console logs only"):
        Visualizer(timing=mode, summary=False)


def test_raw_frames_report_unknown_model_time(image):
    visualizer = Visualizer(timing=True, summary=False)
    visualizer(image)
    visualizer(image)
    assert visualizer.last_timing.frame == 2
    assert visualizer.last_timing.inference_seconds is None


def test_adaptor_preserves_inference_time(image):
    adapted = from_ultralytics(result(image, 12.5))
    assert adapted.inference_seconds == 0.0125


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf"), True])
def test_native_inference_validation(image, seconds):
    with pytest.raises(ValueError, match="inference_seconds"):
        Prediction(image, inference_seconds=seconds)


def test_flat_parameters_and_rgb_text_override(arial):
    image = np.zeros((200, 400, 3), np.uint8)
    visualizer = Visualizer(
        font_size=36,
        label_padding=12,
        show_conf=False,
        text_label_color=(225, 255, 37),
        summary=False,
    )
    output = visualizer(Prediction(image, (Instance(0, name="person", box=(30, 90, 250, 180)),)))
    assert ((output == np.array([37, 255, 225])).all(axis=2)).any()
    assert visualizer.style.font_size == 36
    assert visualizer.style.label_padding == 12
    assert visualizer.style.show_conf is False


@pytest.mark.parametrize("rgb", [(1, 2), (256, 0, 0), (-1, 0, 0), (True, 0, 0), "white"])
def test_invalid_rgb_override(rgb):
    with pytest.raises(ValueError, match="text_label_color"):
        Visualizer(text_label_color=rgb, summary=False)


def test_invalid_timing_options():
    with pytest.raises(ValueError, match="timing must be"):
        Visualizer(timing="unknown", summary=False)
    with pytest.raises(TypeError, match="on_timing must be callable"):
        Visualizer(on_timing=42, summary=False)


def test_process_exposes_flat_timing_and_callback(image):
    records = []
    process(
        Prediction(image, inference_seconds=0.01),
        labels=False,
        on_timing=records.append,
        summary=False,
    )
    assert len(records) == 1
    assert records[0].inference_seconds == 0.01


def test_frame_counter_spans_image_sources_and_annotation_resets(image):
    records = []
    visualizer = Visualizer(labels=False, on_timing=records.append, summary=False)
    first = result(image)
    second = result(image)
    first.path, second.path = "first.jpg", "second.jpg"
    visualizer([first, second])
    assert [record.frame for record in records] == [1, 2]
    visualizer.reset()
    visualizer(first)
    assert visualizer.last_timing.frame == 3
