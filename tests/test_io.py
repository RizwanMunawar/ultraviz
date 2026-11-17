# ruff: noqa: D100, D103
# Scenario names document tests; library definitions require docstrings.
import cv2
import numpy as np
import pytest

from ultraviz import Prediction, Visualizer, save_image, save_video


def test_image_roundtrip(image, tmp_path):
    output = tmp_path / "frame.png"
    save_image(image, output)
    assert np.array_equal(cv2.imread(str(output)), image)
    with pytest.raises(FileExistsError):
        save_image(image, output)
    save_image(image, output, overwrite=True)
    assert not list(tmp_path.glob(".ultraviz-*"))


@pytest.mark.parametrize("extension", [".avi", ".mp4"])
def test_video_roundtrip(image, style, tmp_path, extension):
    output = tmp_path / ("video" + extension)
    summary = save_video(
        (Prediction(image + i * 30) for i in range(4)),
        output,
        fps=12,
        visualizer=Visualizer(style, summary=False),
    )
    assert summary.frames == 4
    assert summary.size == (128, 96)
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        assert len(frames) == 4
        assert frames[-1].mean() == pytest.approx(90, abs=5)
    finally:
        capture.release()


def test_video_failure_preserves_output_and_closes(image, style, tmp_path):
    output = tmp_path / "video.avi"
    output.write_bytes(b"existing video")
    closed = []

    def results():
        try:
            yield Prediction(image)
            yield Prediction(image[:40, :40])
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="dimensions"):
        save_video(
            results(), output, fps=30, visualizer=Visualizer(style, summary=False), overwrite=True
        )
    assert closed == [True]
    assert output.read_bytes() == b"existing video"
    assert not list(tmp_path.glob(".ultraviz-*"))


def test_source_protected(image, style, tmp_path):
    output = tmp_path / "source.avi"
    output.write_bytes(b"input")
    with pytest.raises(ValueError, match="input source"):
        save_video(
            [Prediction(image, source=str(output))],
            output,
            fps=30,
            visualizer=Visualizer(style, summary=False),
            overwrite=True,
        )
    assert output.read_bytes() == b"input"


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps(image, tmp_path, fps):
    with pytest.raises(ValueError, match="fps"):
        save_video([Prediction(image)], tmp_path / "out.mp4", fps=fps)


def test_empty_video(style, tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_video([], tmp_path / "out.avi", fps=30, visualizer=Visualizer(style, summary=False))
    assert not list(tmp_path.iterdir())


def test_odd_dimensions(image, style, tmp_path):
    with pytest.raises(ValueError, match="even"):
        save_video(
            [Prediction(image[:, :127])],
            tmp_path / "out.mp4",
            fps=30,
            visualizer=Visualizer(style, summary=False),
        )
