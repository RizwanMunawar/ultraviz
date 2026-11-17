# ruff: noqa: D103
"""Public Visualizer configuration, state reuse and result conversion coverage."""

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

from ultraviz import Instance, Prediction, Visualizer, from_ultralytics


@pytest.mark.parametrize(
    "args", [{"line_width": 4, "labels": False}, Namespace(line_width=4, labels=False)]
)
def test_dictionary_and_namespace_are_not_modified(args, image):
    before = vars(args).copy() if isinstance(args, Namespace) else args.copy()
    visualizer = Visualizer(args, line_width=3)
    assert visualizer.style.line_width == 3
    assert visualizer.style.labels is False
    assert (vars(args) if isinstance(args, Namespace) else args) == before
    assert visualizer(Prediction(image)).shape == image.shape


def test_reuses_cached_font_between_calls(image, arial):
    result = Prediction(image, (Instance(0, name="person", box=(20, 30, 80, 80)),))
    visualizer = Visualizer(font_size=18)
    visualizer(result)
    font = visualizer._annotator._font
    visualizer(result)
    assert visualizer._annotator._font is font


@pytest.mark.parametrize(
    "tracked,obb", [(False, False), (True, False), (False, True), (True, True)]
)
def test_one_cpu_transfer_for_box_metadata(image, tracked, obb):
    transfers = []
    row = [50.0, 50.0, 40.0, 20.0, 0.5] if obb else [20.0, 20.0, 80.0, 80.0]
    if tracked:
        row.append(42.0)
    row.extend([0.9, 0.0])

    class Tensor:
        """Tensor double that counts transfers without requiring a GPU."""

        def detach(self):
            """Return detached metadata."""
            return self

        def cpu(self):
            """Record one device transfer."""
            transfers.append("cpu")
            return self

        def numpy(self):
            """Return the box metadata as a NumPy batch."""
            return np.array([row])

    box_data = SimpleNamespace(data=Tensor())
    if obb:
        box_data.xyxyxyxy = np.array([[[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]]])
    result = SimpleNamespace(
        orig_img=image, names={0: "person"}, **{"obb" if obb else "boxes": box_data}
    )
    instance = from_ultralytics(result).instances[0]
    assert transfers == ["cpu"]
    assert instance.track_id == (42 if tracked else None)
    assert instance.confidence == 0.9
    assert instance.class_id == 0


def test_disabled_layers_skip_tensor_conversion_for_render_and_export(image, tmp_path):
    class UnavailableTensor:
        """Tensor double that fails if a disabled layer is copied to CPU."""

        def cpu(self):
            """Reject unexpected transfers for disabled annotation layers."""
            raise AssertionError("Disabled layers must not transfer data")

    result = SimpleNamespace(
        orig_img=image,
        names={0: "person"},
        path="input.jpg",
        boxes=SimpleNamespace(data=np.array([[20.0, 20.0, 80.0, 80.0, 0.9, 0.0]])),
        masks=SimpleNamespace(data=UnavailableTensor()),
        keypoints=SimpleNamespace(data=UnavailableTensor()),
        semantic_mask=SimpleNamespace(data=UnavailableTensor()),
    )
    options = {"masks": False, "pose": False, "labels": False}
    assert Visualizer(options)(result).any()
    summary = Visualizer(options, save=tmp_path / "frame.png")(result)
    assert summary.frames == 1
    summary = Visualizer(options, save=tmp_path / "video.avi")([result])
    assert summary.frames == 1
