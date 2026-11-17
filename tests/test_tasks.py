# ruff: noqa: D103
"""Task selection validates model compatibility before executing inference."""

from types import SimpleNamespace

import pytest

from ultraviz import Visualizer, predict
from ultraviz.inference import TASKS


def make_model(task, calls):
    return SimpleNamespace(
        task=task,
        predict=lambda **options: calls.append(("predict", options)) or iter(["result"]),
        track=lambda **options: calls.append(("track", options)) or iter(["tracked"]),
    )


@pytest.mark.parametrize("task", [task for task in TASKS if task != "track"])
def test_each_task_uses_matching_model(task):
    calls = []
    assert list(predict(make_model(task, calls), "image.jpg", task=task)) == ["result"]
    assert calls == [("predict", {"source": "image.jpg", "stream": True})]


@pytest.mark.parametrize("task", ["detect", "segment", "pose", "obb"])
def test_track_mode_and_explicit_task_combination(task):
    for options in ({"task": "track"}, {"task": task, "tracking": True}):
        calls = []
        assert list(predict(make_model(task, calls), "video.mp4", **options)) == ["tracked"]
        assert calls[0][0] == "track"


def test_mismatched_weights_do_not_run_prediction():
    calls = []
    with pytest.raises(ValueError, match="matching weights"):
        predict(make_model("detect", calls), "image.jpg", task="segment")
    assert calls == []


@pytest.mark.parametrize("task", ["classify", "semantic", "depth"])
def test_tracking_rejects_incompatible_models(task):
    calls = []
    with pytest.raises(ValueError, match="Tracking"):
        predict(make_model(task, calls), "image.jpg", task="track")
    assert not calls


def test_visualizer_source_task_is_lazy_and_failure_closes_stream():
    calls = []
    source = Visualizer("video.mp4", model=make_model("detect", calls), task="segment")
    assert not calls
    with pytest.raises(ValueError, match="matching weights"):
        next(source)
    with pytest.raises(StopIteration):
        next(source)
    assert not calls


def test_nested_task_and_conflicting_settings():
    calls = []
    source = Visualizer("video.mp4", model=make_model("pose", calls), inference={"task": "pose"})
    assert next(source) == "result"
    source.close()
    with pytest.raises(ValueError, match="Conflicting task"):
        Visualizer(
            "video.mp4",
            model=make_model("detect", calls),
            task="detect",
            inference={"task": "segment"},
        )


def test_invalid_task_and_inference_type_fail_before_loading():
    with pytest.raises(ValueError, match="task must be one of"):
        Visualizer("video.mp4", model="unused.pt", task="typo")
    with pytest.raises(TypeError, match="inference must be a mapping"):
        Visualizer("video.mp4", model="unused.pt", inference="incorrect")


def test_task_selection_in_argument_dictionary():
    calls = []
    source = Visualizer(
        {"task": "segment", "tracking": True},
        source="video.mp4",
        model=make_model("segment", calls),
    )
    assert next(source) == "tracked"
    source.close()


@pytest.mark.parametrize("key", ["source", "model", "tracking"])
def test_reserved_inference_arguments_fail_early(key):
    with pytest.raises(ValueError, match="as Visualizer arguments"):
        Visualizer("video.mp4", model="unused.pt", inference={key: "incorrect"})
