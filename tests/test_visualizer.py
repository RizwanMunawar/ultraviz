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


@pytest.mark.parametrize(
    "height,width,line,font",
    [
        (360, 640, 1, 10),
        (720, 1280, 3, 20),
        (1080, 1920, 4, 30),
        (1440, 2560, 5, 40),
        (2160, 3840, 8, 60),
        (4320, 7680, 16, 120),
    ],
)
def test_automatic_sizes_follow_resolution_in_both_orientations(height, width, line, font):
    style = Visualizer().style
    for shape in ((height, width), (width, height)):
        assert style.resolved_line_width(shape) == line
        assert style.resolved_font_size(shape) == font


def test_portrait_and_landscape_render_with_same_font_and_outline(arial, monkeypatch):
    widths = []
    monkeypatch.setattr(
        "ultraviz.annotation.Annotator._outline",
        lambda self, image, path, bgr, line: widths.append(line),
    )
    visualizer = Visualizer(show_conf=False)
    sizes = []
    for shape in ((720, 1280), (1280, 720), (1080, 1920), (1920, 1080)):
        height, width = shape
        image = np.zeros((height, width, 3), np.uint8)
        item = Instance(0, name="person", box=(100, 100, width - 100, height - 100))
        output = visualizer(Prediction(image, (item,)))
        sizes.append(visualizer._annotator._font.size)
        assert output.shape == image.shape
    assert widths == [3, 3, 4, 4]
    assert sizes == [20, 20, 30, 30]


def test_fixed_overrides_and_tiny_object_limits_on_large_images():
    automatic = Visualizer().style
    fixed = Visualizer(line_width=1, font_size=20).style
    for shape in ((720, 1280), (1080, 1920), (7680, 4320)):
        assert fixed.resolved_line_width(shape) == 1
        assert fixed.resolved_font_size(shape) == 20
        assert automatic.resolved_line_width(shape, (12, 20)) == 1


@pytest.mark.parametrize("shape", [(1080, 1920), (1920, 1080)])
def test_small_and_large_object_labels_keep_default_thirty_pixel_font(shape, arial, monkeypatch):
    from PIL import ImageFont

    font_sizes = []
    original = ImageFont.FreeTypeFont.getbbox

    def record(font, text, *args, **kwargs):
        font_sizes.append(font.size)
        return original(font, text, *args, **kwargs)

    monkeypatch.setattr(ImageFont.FreeTypeFont, "getbbox", record)
    image = np.zeros((*shape, 3), np.uint8)
    result = Prediction(
        image,
        (
            Instance(0, name="small", box=(100, 100, 112, 120)),
            Instance(1, name="medium", box=(200, 200, 250, 260)),
            Instance(2, name="large", box=(400, 300, 700, 600)),
        ),
    )
    visualizer = Visualizer(show_conf=False)
    frame = visualizer.annotate(result)
    assert font_sizes and set(font_sizes) == {30}
    assert visualizer._annotator._font.getname()[0] == "Arial"
    assert frame.any()
    assert not image.any()
