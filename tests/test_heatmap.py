# ruff: noqa: D103
"""Density shading remains smooth, bounded, branded and independent of mask rendering."""

import numpy as np
import pytest

from ultraviz import Instance, Prediction, Visualizer
from ultraviz.heatmap import heatmap_lut


def test_hotspot_is_smooth_and_inactive_scene_pixels_are_unchanged():
    image = np.full((360, 640, 3), 80, np.uint8)
    prediction = Prediction(image, (Instance(0, box=(310, 170, 330, 190)),))
    output = Visualizer(
        demo="heatmap",
        summary=False,
        labels=False,
        boxes=False,
        heatmap_radius=40,
        heatmap_contours=False,
    )(prediction)
    np.testing.assert_array_equal(output[:40], image[:40])
    assert not np.array_equal(output[180, 320], image[180, 320])
    assert len(np.unique(output[180, 200:440], axis=0)) > 60
    np.testing.assert_array_equal(image, 80)


def test_heatmap_has_its_own_opacity_and_respects_zero():
    image = np.full((96, 128, 3), 80, np.uint8)
    result = Prediction(image, (Instance(0, box=(30, 30, 70, 70)),))
    options = dict(demo="heatmap", summary=False, labels=False, boxes=False, mask_opacity=0)
    np.testing.assert_array_equal(Visualizer(**options, heatmap_opacity=0)(result), image)
    assert not np.array_equal(Visualizer(**options)(result), image)


def test_empty_heatmap_never_tints_the_source():
    image = np.full((96, 128, 3), 80, np.uint8)
    np.testing.assert_array_equal(
        Visualizer(demo="heatmap", labels=False, summary=False)(Prediction(image)), image
    )


def test_radius_controls_footprint_in_image_pixels():
    image = np.zeros((360, 640, 3), np.uint8)
    prediction = Prediction(image, (Instance(0, box=(310, 170, 330, 190)),))
    options = dict(demo="heatmap", summary=False, boxes=False, labels=False, heatmap_contours=False)
    small = Visualizer(**options, heatmap_radius=15)(prediction)
    broad = Visualizer(**options, heatmap_radius=60)(prediction)
    assert np.count_nonzero(broad) > 3 * np.count_nonzero(small)


def test_landscape_and_portrait_have_equivalent_density_footprints():
    image = np.zeros((360, 640, 3), np.uint8)
    options = dict(demo="heatmap", summary=False, boxes=False, labels=False, heatmap_contours=False)
    result = Visualizer(**options)(Prediction(image, (Instance(0, box=(310, 170, 330, 190)),)))
    rotated = Visualizer(**options)(
        Prediction(image.transpose(1, 0, 2).copy(), (Instance(0, box=(170, 310, 190, 330)),))
    )
    # OpenCV integer resize can round transposed interpolation by one level.
    np.testing.assert_allclose(rotated.transpose(1, 0, 2), result, atol=1, rtol=0)


def test_heatmap_panel_appears_without_summary_flag(arial):
    image = np.full((360, 640, 3), 80, np.uint8)
    visualizer = Visualizer(demo="heatmap", labels=False, boxes=False, heatmap_opacity=0)
    output = visualizer(Prediction(image))
    assert visualizer.last_summary.task == "heatmap"
    assert not np.array_equal(output[30:130, -220:-30], image[30:130, -220:-30])
    np.testing.assert_array_equal(output[200:], image[200:])


def test_contours_are_subtle_and_can_be_disabled():
    image = np.full((240, 400, 3), 80, np.uint8)
    prediction = Prediction(image, (Instance(0, box=(180, 100, 220, 140)),))
    options = dict(demo="heatmap", summary=False, boxes=False, labels=False)
    smooth = Visualizer(**options, heatmap_contours=False)(prediction)
    contoured = Visualizer(**options, heatmap_contours=True)(prediction)
    difference = np.abs(contoured.astype(int) - smooth.astype(int))
    assert 0 < difference.max() < 40


@pytest.mark.parametrize(
    "options",
    [
        {"heatmap_opacity": -1},
        {"heatmap_opacity": float("nan")},
        {"heatmap_radius": 0},
        {"heatmap_radius": float("inf")},
        {"heatmap_radius": True},
        {"heatmap_contours": "yes"},
    ],
)
def test_invalid_heatmap_options_fail_early(options):
    with pytest.raises(ValueError):
        Visualizer(**options)


def test_color_ramp_is_immutable_and_shared():
    lut = heatmap_lut()
    assert lut is heatmap_lut()
    assert not lut.flags.writeable
    assert lut.shape == (256, 1, 3)


def test_heatmap_defaults_to_density_only_and_explicit_layers_override():
    visualizer = Visualizer(demo="heatmap")
    assert visualizer.style.summary is True
    assert not any(
        (
            visualizer.style.boxes,
            visualizer.style.labels,
            visualizer.style.masks,
            visualizer.style.pose,
            visualizer.style.trails,
        )
    )
    explicit = Visualizer(demo="heatmap", boxes=True, labels=True)
    assert explicit.style.boxes and explicit.style.labels
