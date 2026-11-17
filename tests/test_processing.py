# ruff: noqa: D103
# Scenario names document tests; library definitions require docstrings.
"""One-call processing: exports, preview lifecycle and stream ownership."""

import cv2
import pytest

from ultraviz import Instance, Prediction, ProcessSummary, Visualizer, process


@pytest.fixture(params=["function", "class"])
def processor(request):
    if request.param == "function":
        return process
    return lambda results, **options: Visualizer(options)(results)


def predictions(image, count=3):
    for i in range(count):
        yield Prediction(image, (Instance(0, box=(10 + i * 20, 20, 30 + i * 20, 40), track_id=7),))


def test_direct_style_options_and_lazy_tracking(processor, image):
    frames = processor(predictions(image), labels=False, boxes=False, trail_length=2)
    assert not next(frames).any()
    assert next(frames)[30, 30].any()
    assert next(frames)[30, 50].any()
    frames.close()


def test_save_image_singleton_result_list(processor, image, tmp_path):
    target = tmp_path / "nested" / "annotated.png"
    summary = processor(list(predictions(image, 1)), save=target, labels=False)
    assert summary == ProcessSummary(1, target)
    assert cv2.imread(str(target)).any()


def test_save_video_stream_and_close(processor, image, tmp_path):
    events = []

    def results():
        try:
            yield from predictions(image)
        finally:
            events.append("closed")

    target = tmp_path / "nested" / "annotated.avi"
    summary = processor(results(), save=target, fps=12, labels=False)
    assert summary == ProcessSummary(3, target)
    assert events == ["closed"]
    capture = cv2.VideoCapture(str(target))
    try:
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 3
        assert capture.get(cv2.CAP_PROP_FPS) == 12
    finally:
        capture.release()


def test_directory_export(processor, image, tmp_path):
    target = tmp_path / "frames"
    summary = processor(predictions(image), save=target, labels=False)
    assert summary == ProcessSummary(3, target)
    assert [p.name for p in sorted(target.iterdir())] == [
        "frame_000001.png",
        "frame_000002.png",
        "frame_000003.png",
    ]


def test_image_path_rejects_multiple_results_before_writing(processor, image, tmp_path):
    target = tmp_path / "frame.png"
    with pytest.raises(ValueError, match="Multiple results"):
        processor(predictions(image), save=target, labels=False)
    assert not target.exists()


def test_protect_image_source(processor, image, tmp_path):
    target = tmp_path / "frame.png"
    target.write_bytes(b"source")
    with pytest.raises(ValueError, match="input source"):
        processor(Prediction(image, source=str(target)), save=target, overwrite=True, labels=False)
    assert target.read_bytes() == b"source"


def test_show_stop_and_cleanup(processor, image, monkeypatch):

    import ultraviz.preview as gui

    events = []
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(gui.cv2, "namedWindow", lambda *args: events.append("open"))
    monkeypatch.setattr(gui.cv2, "imshow", lambda *args: events.append("show"))
    monkeypatch.setattr(gui.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(gui.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(gui.cv2, "destroyWindow", lambda *args: events.append("destroy"))

    def results():
        try:
            yield from predictions(image)
        finally:
            events.append("close results")

    assert processor(results(), show=True, labels=False) == ProcessSummary(1, None, True)
    assert events == ["open", "show", "destroy", "close results"]


def test_preview_exception_preserves_existing_video(processor, image, tmp_path, monkeypatch):
    import ultraviz.processing as api

    target = tmp_path / "out.avi"
    target.write_bytes(b"original")
    monkeypatch.setattr(
        api.PreviewWindow,
        "__call__",
        lambda *args: (_ for _ in ()).throw(RuntimeError("GUI failure")),
    )
    with pytest.raises(RuntimeError, match="GUI failure"):
        processor(predictions(image), show=True, save=target, overwrite=True, labels=False)
    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob(".ultraviz-*"))


def test_show_and_save_stop_exports_only_processed_frames(processor, image, tmp_path, monkeypatch):
    import ultraviz.processing as api

    def stop(self, frame):
        self.stopped = True
        return False

    monkeypatch.setattr(api.PreviewWindow, "__call__", stop)
    target = tmp_path / "out.avi"
    summary = processor(predictions(image), show=True, save=target, labels=False)
    assert summary == ProcessSummary(1, target, True)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"font_sze": 20}, TypeError),
        ({"save": True}, TypeError),
        ({"show": "yes"}, TypeError),
        ({"fps": 0}, ValueError),
        ({"fps": float("nan")}, ValueError),
        ({"mask_opacity": 2}, ValueError),
    ],
)
def test_bad_options_fail_before_consuming(processor, image, options, error):
    events = []

    def results():
        events.append("consumed")
        yield Prediction(image)

    with pytest.raises(error):
        processor(results(), **options)
    assert not events


def test_style_overrides(processor, image, style):
    result = next(predictions(image, 1))
    assert not process(result, style=style, boxes=False).any()
