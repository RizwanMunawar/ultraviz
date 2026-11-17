# ruff: noqa: D103
"""Source inference stays lazy and previews frames before requesting the next."""

import numpy as np
import pytest

from ultraviz import Instance, Prediction, Visualizer, process


class Model:
    """Model double that records calls and closes its sequential result stream."""

    def __init__(self, image, events):
        """Keep a frame and an event log without performing inference."""
        self.image, self.events = image, events
        self.options = None

    def predict(self, **options):
        """Return one result per pull so tests can verify display ordering."""
        self.options = options
        self.events.append("predict")

        def results():
            """Record each inferred frame and the release of upstream resources."""
            try:
                for i in range(3):
                    self.events.append(f"infer {i}")
                    yield Prediction(
                        self.image, (Instance(0, name="person", box=(20, 20, 80, 80)),)
                    )
            finally:
                self.events.append("closed")

        return results()

    def track(self, **options):
        """Reuse the lazy prediction stream for tracking argument checks."""
        self.events.append("track")
        return self.predict(**options)


def mock_preview(monkeypatch, events):
    import ultraviz.preview as gui

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(gui.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(gui.cv2, "imshow", lambda *args: events.append("show"))
    monkeypatch.setattr(gui.cv2, "waitKey", lambda *args: -1)
    monkeypatch.setattr(gui.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(gui.cv2, "destroyWindow", lambda *args: events.append("destroy"))


def test_source_is_lazy_and_exposes_raw_results(image):
    events = []
    model = Model(image, events)
    source = Visualizer(
        "1.mp4", model=model, labels=False, inference={"device": "cpu", "imgsz": 320}
    )
    assert not events
    result = next(source)
    assert source.last_result is result
    assert isinstance(result, Prediction)
    assert model.options == {
        "source": "1.mp4",
        "stream": True,
        "verbose": False,
        "batch": 1,
        "stream_buffer": False,
        "device": "cpu",
        "imgsz": 320,
    }
    source.close()
    assert events[-1] == "closed"
    with pytest.raises(StopIteration):
        next(source)


@pytest.mark.parametrize("direct", [True, False])
def test_source_preview_interleaves_inference_and_display(image, monkeypatch, direct):
    events = []
    mock_preview(monkeypatch, events)
    source = Visualizer("1.mp4", model=Model(image, events), show=True, labels=False)
    summary = source() if direct else process(source, show=True)
    assert summary.frames == 3
    assert events == [
        "predict",
        "infer 0",
        "show",
        "infer 1",
        "show",
        "infer 2",
        "show",
        "closed",
        "destroy",
    ]


def test_source_callback_extracts_each_result(image):
    extracted = []
    source = Visualizer(
        "image.jpg", model=Model(image, []), labels=False, on_result=extracted.append, tracking=True
    )
    raw = list(source)
    assert extracted == raw
    assert source.last_result is raw[-1]


def test_process_inherits_source_style(image):
    source = Visualizer("image.jpg", model=Model(image, []), labels=False, boxes=False)
    frame = next(process(source))
    assert not frame.any()
    source.close()


def test_source_requires_explicit_model():
    with pytest.raises(ValueError, match="model=model"):
        Visualizer("1.mp4")


def test_inference_rejects_stream_override(image):
    with pytest.raises(ValueError, match="always streams"):
        Visualizer("1.mp4", model=Model(image, []), inference={"stream": False})


def test_source_early_stop_closes_inference(image, monkeypatch):
    events = []
    mock_preview(monkeypatch, events)
    monkeypatch.setattr("ultraviz.preview.cv2.waitKey", lambda *args: ord("q"))
    source = Visualizer("1.mp4", model=Model(image, events), labels=False)
    summary = process(source, show=True)
    assert summary.frames == 1 and summary.stopped
    assert "infer 1" not in events
    assert events[-1] == "closed"


def test_annotated_frames_can_be_previewed_and_exported(image, monkeypatch, tmp_path):
    events = []
    mock_preview(monkeypatch, events)
    summary = process(iter([image, image]), show=True, save=tmp_path / "frames.avi")
    assert summary.frames == 2
    assert events.count("show") == 2
    assert np.array_equal(process(image), image)


def test_preview_does_not_add_full_frame_delay_after_slow_inference(image, monkeypatch):
    import ultraviz.preview as gui

    events = []
    mock_preview(monkeypatch, events)
    times = iter([0.0, 0.001, 0.201, 0.202])
    monkeypatch.setattr(gui, "perf_counter", lambda: next(times))
    delays = []
    monkeypatch.setattr(gui.cv2, "waitKey", lambda delay: delays.append(delay) or -1)
    preview = gui.PreviewWindow(True, 30)
    assert preview(image)
    assert preview(image)
    assert delays == [1, 1]
    preview.close()
