# ruff: noqa: D103
# Scenario names document tests; library definitions require docstrings.
"""Actual Ultralytics Results on CPU, without model downloads."""

import numpy as np
import pytest

from ultraviz import Visualizer, from_ultralytics, predict, process


def test_real_results_all_tasks(image, style):
    torch = pytest.importorskip("torch")
    results = pytest.importorskip("ultralytics.engine.results")
    common = dict(orig_img=image, path="frame.jpg", names={0: "person", 1: "car"})
    boxes = torch.tensor([[20.0, 20.0, 70.0, 70.0, 0.9, 0.0]])
    tracked = torch.tensor([[20.0, 20.0, 70.0, 70.0, 42.0, 0.9, 0.0]])
    mask = torch.zeros((1, 96, 128))
    mask[:, 20:70, 20:70] = 1
    keypoints = torch.tensor([[[30.0, 30.0, 0.9], [60.0, 60.0, 0.9]]])
    obb = torch.tensor([[60.0, 40.0, 40.0, 20.0, 0.6, 0.9, 0.0]])
    tracked_obb = torch.tensor([[60.0, 40.0, 40.0, 20.0, 0.6, 42.0, 0.9, 0.0]])
    fixtures = [
        results.Results(**common, boxes=boxes),
        results.Results(**common, boxes=tracked),
        results.Results(**common, boxes=boxes, masks=mask),
        results.Results(**common, boxes=tracked, masks=mask),
        results.Results(**common, boxes=boxes, keypoints=keypoints),
        results.Results(**common, boxes=tracked, keypoints=keypoints),
        results.Results(**common, obb=obb),
        results.Results(**common, obb=tracked_obb),
        results.Results(**common, probs=torch.tensor([0.2, 0.8])),
        results.Results(**common, boxes=torch.empty((0, 6))),
    ]
    renderer = Visualizer(style)
    for result in fixtures:
        output = renderer(result)
        assert output.shape == image.shape
        assert output.dtype == np.uint8
    for frames in (
        renderer(fixtures),
        renderer(iter(fixtures)),
        process(fixtures, style=style),
        process(iter(fixtures), style=style),
    ):
        frames = list(frames)
        assert len(frames) == len(fixtures)
        assert all(frame.shape == image.shape for frame in frames)
    assert from_ultralytics(fixtures[1]).instances[0].track_id == 42
    assert from_ultralytics(fixtures[3]).instances[0].track_id == 42
    assert from_ultralytics(fixtures[5]).instances[0].track_id == 42
    assert from_ultralytics(fixtures[7]).instances[0].track_id == 42


def test_model_protocol():
    class Model:
        """Model double for checking inference argument forwarding."""

        def predict(self, **options):
            """Check prediction arguments and yield a sentinel result."""
            assert options == {"source": "input.mp4", "stream": True, "conf": 0.3}
            return iter(["predict"])

        def track(self, **options):
            """Check streaming tracking and yield a sentinel result."""
            assert options["stream"] is True
            return iter(["track"])

    assert list(predict(Model(), "input.mp4", conf=0.3)) == ["predict"]
    assert list(predict(Model(), "input.mp4", tracking=True)) == ["track"]
    with pytest.raises(ValueError, match="stream"):
        predict(Model(), "input.mp4", stream=False)


def test_real_dense_results(image, style):
    import inspect

    torch = pytest.importorskip("torch")
    results = pytest.importorskip("ultralytics.engine.results")
    if "semantic_mask" not in inspect.signature(results.Results).parameters:
        pytest.skip("Installed Ultralytics does not expose dense Results yet")
    common = dict(orig_img=image, path="dense.jpg", names={0: "background", 1: "road"})
    semantic = torch.ones(image.shape[:2], dtype=torch.int64)
    depth = torch.ones(image.shape[:2])
    for result in (
        results.Results(**common, semantic_mask=semantic),
        results.Results(**common, depth=depth),
    ):
        renderer = Visualizer(style)
        assert renderer(result).any()
        assert process(result, style=style).any()
        assert next(process(iter([result]), style=style)).any()
        assert renderer([result])[0].any()
        assert next(renderer(iter([result]))).any()
