# ruff: noqa: D103
"""Regressions for mask optimization, cleanup and validated native data."""

import numpy as np
import pytest

from ultraviz import Instance, Prediction, Style, Visualizer, process, save_video
from ultraviz.geometry import rounded_rectangle


@pytest.mark.parametrize("opacity", [0.0, 0.33, 0.45, 1.0])
@pytest.mark.parametrize("layout", ["sparse", "empty", "full", "fortran"])
def test_mask_region_optimization_matches_full_frame_reference(image, opacity, layout):
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, image.shape, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], bool)
    if layout == "full":
        mask[:] = True
    elif layout != "empty":
        mask[10:70, 20:90] = True
        mask[20:40, 30:60] = False
    if layout == "fortran":
        mask = np.asfortranarray(mask)
    expected = image.copy()
    expected[mask] = np.rint(image[mask] * (1 - opacity) + np.array([255, 42, 4]) * opacity)
    output = Visualizer(labels=False, boxes=False, mask_opacity=opacity)(
        Prediction(image, (Instance(0, mask=mask),))
    )
    assert np.array_equal(expected, output)


def test_mask_copies_source_once_and_remains_isolated(image):
    mask = np.zeros(image.shape[:2], bool)
    instance = Instance(0, mask=mask)
    mask[:] = True
    assert not instance.mask.any()
    assert not instance.mask.flags.writeable


def test_boolean_options_and_missing_class_id_are_rejected():
    with pytest.raises(ValueError, match="class_id"):
        Instance(None)
    with pytest.raises(ValueError, match="labels must be a boolean"):
        Style(labels="False")


def test_classification_metadata_is_frozen(image):
    items = [["cat", 0.8]]
    prediction = Prediction(image, classifications=items)
    items[0][0] = "dog"
    assert prediction.classifications == (("cat", 0.8),)


def test_extreme_finite_depth_does_not_overflow(image):
    depth = np.zeros(image.shape[:2])
    depth[0, 0], depth[0, 1] = -1e308, 1e308
    with np.errstate(all="raise"):
        output = Visualizer(labels=False)(Prediction(image, depth=depth))
    assert output.any()


def test_rounded_rectangle_cache_cannot_be_modified():
    path = rounded_rectangle(40, 20, 6)
    assert not path.flags.writeable
    first = path.copy()
    translated = path + [10, 20]
    assert np.array_equal(rounded_rectangle(40, 20, 6), first)
    assert np.array_equal(translated - [10, 20], first)


def test_preview_cleanup_error_still_closes_input(image, monkeypatch):
    events = []
    monkeypatch.setattr("ultraviz.processing.PreviewWindow.__call__", lambda *args: False)

    def fail(*args):
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr("ultraviz.processing.PreviewWindow.close", fail)

    def results():
        try:
            yield Prediction(image)
        finally:
            events.append("closed")

    with pytest.raises(RuntimeError, match="cleanup failure"):
        process(results(), show=True, labels=False)
    assert events == ["closed"]


def test_process_reuses_source_annotation_engine(image):
    from types import SimpleNamespace

    source = Visualizer(
        "source.mp4",
        model=SimpleNamespace(predict=lambda **kwargs: iter([Prediction(image)])),
        labels=False,
    )
    source._annotator.marker = "original engine"
    frames = process(source)
    next(frames)
    assert source._annotator._frame == 1
    frames.close()


def test_video_rejects_boolean_frame_rate(image, tmp_path):
    with pytest.raises(ValueError, match="fps"):
        save_video([Prediction(image)], tmp_path / "out.avi", fps=True)


def test_large_label_sprites_have_a_byte_budget(arial):
    image = np.zeros((300, 4000, 3), np.uint8)
    visualizer = Visualizer(font_size=64, confidence=False, boxes=False)
    for i in range(20):
        visualizer(Prediction(image, (Instance(0, name=f"{i}:" + "x" * 60),)))
    engine = visualizer._annotator
    assert engine._label_cache_bytes <= 8 * 1024 * 1024
    assert engine._label_cache_bytes == sum(
        bgr.nbytes + mask.nbytes for bgr, mask in engine._label_cache.values()
    )


@pytest.mark.parametrize("field", ["depth", "semantic"])
def test_dense_maps_reject_complex_values(image, field):
    with pytest.raises(ValueError, match="real numeric"):
        Prediction(image, **{field: np.ones(image.shape[:2], dtype=complex)})
