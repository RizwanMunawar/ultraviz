# ruff: noqa: D103
"""Tracking text, shared label spacing and bounded sequence state."""

import numpy as np
import pytest

from ultraviz import Instance, Prediction, Style, Visualizer
from ultraviz.annotation import Annotator


@pytest.mark.parametrize(
    "geometry",
    [
        {"box": (20, 30, 80, 80)},
        {"corners": np.array([[20, 30], [80, 30], [80, 80], [20, 80]])},
        {"keypoints": np.array([[20, 30], [80, 80]])},
        {"mask": np.ones((96, 128), bool)},
    ],
)
@pytest.mark.parametrize("track_id", [0, 42])
def test_tracked_instances_display_only_id(image, geometry, track_id, monkeypatch):
    captured = []

    def capture(self, frame, labels):
        captured.extend(text for text, *_ in labels)
        return frame

    monkeypatch.setattr(Annotator, "_labels", capture)
    item = Instance(0, name="person", confidence=0.87, track_id=track_id, **geometry)
    Visualizer()(Prediction(image, (item,)))
    assert captured == [str(track_id)]


def test_untracked_labels_keep_name_and_confidence(image, monkeypatch):
    captured = []
    monkeypatch.setattr(
        Annotator,
        "_labels",
        lambda self, frame, labels: captured.extend(text for text, *_ in labels) or frame,
    )
    Visualizer()(Prediction(image, (Instance(0, name="person", confidence=0.87),)))
    assert captured == ["person 0.87"]


def test_track_ids_false_hides_label_without_loading_font(image, monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Hidden tracking labels must not load fonts")

    monkeypatch.setattr("ultraviz.annotation.load_font", reject)
    item = Instance(0, name="person", track_id=42, box=(20, 30, 80, 80))
    assert Visualizer(track_ids=False)(Prediction(image, (item,))).any()


def test_padding_increases_both_dimensions_and_margin_separates_label(arial):
    image = np.zeros((200, 400, 3), np.uint8)
    item = Prediction(image, (Instance(0, name="person", box=(60, 120, 180, 180)),))
    sizes = []
    for padding in (2, 10):
        visualizer = Visualizer(
            boxes=False, confidence=False, font_size=20, label_padding=padding, label_margin=7
        )
        output = visualizer(item)
        rows, columns = np.nonzero(output.any(axis=2))
        assert rows.max() == 120 - 7 - 1
        assert columns.min() == 60
        sprite, _ = next(iter(visualizer._annotator._label_cache.values()))
        sizes.append(sprite.shape[:2])
    assert np.array_equal(np.subtract(sizes[1], sizes[0]), (16, 16))


@pytest.mark.parametrize("box", [(0, 0, 80, 80), (180, 170, 199, 199)])
def test_label_margin_preserves_frame_inset(arial, box):
    image = np.zeros((200, 200, 3), np.uint8)
    item = Prediction(image, (Instance(0, name="person", box=box),))
    output = Visualizer(boxes=False, label_padding=8, label_margin=5)(item)
    rows, columns = np.nonzero(output.any(axis=2))
    assert rows.min() >= 5 and rows.max() < 195
    assert columns.min() >= 5 and columns.max() < 195


def test_classification_labels_have_padding_and_separate_rows(arial):
    image = np.zeros((400, 400, 3), np.uint8)
    prediction = Prediction(image, classifications=(("cat", 0.8), ("dog", 0.2)))
    output = Visualizer(font_size=20, label_padding=10, label_margin=6)(prediction)
    occupied = np.flatnonzero(output.any(axis=(1, 2)))
    gaps = np.diff(occupied)
    assert gaps.max() >= 7  # At least six blank rows between labels.
    assert occupied.min() >= 6


@pytest.mark.parametrize(
    "option,value",
    [
        ("label_padding", -1),
        ("label_padding", True),
        ("label_margin", -1),
        ("label_margin", 1.5),
    ],
)
def test_spacing_validation(option, value):
    with pytest.raises(ValueError, match=option):
        Style(**{option: value})


def test_many_tracking_ids_keep_cache_and_history_bounded(image, arial):
    visualizer = Visualizer(font_size=12, trail_length=10, trail_max_gap=5)
    for track_id in range(600):
        prediction = Prediction(
            image, (Instance(0, track_id=track_id, box=(20, 40, 80, 80)),), source="sequence.mp4"
        )
        visualizer(prediction)
    engine = visualizer._annotator
    assert len(engine._label_cache) <= 256
    assert len(engine._tracks) <= 6
    assert all(len(points) <= 10 for _, points in engine._tracks.values())
