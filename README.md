# Ultraviz

Visualize detection, tracking, instance segmentation, pose, oriented boxes,
classification, semantic segmentation and depth results with Ultralytics colors
and Arial annotations. Images and videos use the same API.

## Install locally

Requires Python 3.10 or newer. From the repository directory:

```bash
pip install -e '.[ultralytics]'
```

Use `pip install -e .` when supplying predictions from another backend. The base
package requires NumPy, OpenCV and Pillow. Ultraviz does not include model weights.

## Quick start

```python
from ultralytics import YOLO
from ultraviz import Visualizer

visualizer = Visualizer(
    "input.mp4",  # An image path, video path, URL or camera index.
    model=YOLO("yolo26n.pt"),
    demo="detect",
    summary_title="Traffic overview",
    summary_subtitle="Live vehicle detections",
    show_conf=False,
    show=True,
)
visualizer()
```

Pass visualization arguments directly to `Visualizer`. The model loads once;
source predictions stream one frame at a time. Preview requires a graphical
desktop. Press Q or Escape to stop; any key closes the final frame.

To visualize existing results:

```python
model = YOLO("yolo26n.pt")
results = model("input.mp4", stream=True)
visualizer = Visualizer(
    summary_title="Scene overview",
    summary_subtitle="Camera 01",
    show=True,
)
visualizer(results)
```

Use `stream=True` for videos so frames can display as inference proceeds.
An individual result returns one annotated BGR image; a list returns a list;
an iterator returns lazy annotated frames. With `show` or `save`, the call
consumes results and returns `ProcessSummary(frames, output, stopped)`.

## Tasks and demos

| `demo` | Visualization | Default summary |
| --- | --- | --- |
| `detect` | Rounded boxes and labels | Objects and classes |
| `track` | Numeric IDs and trails | Objects, classes and visible tracks |
| `segment` | Instance masks and boxes | Instances, masks and union coverage |
| `pose` | Keypoints and skeleton | Poses and visible keypoints |
| `obb` | Rounded oriented boxes | Oriented boxes and classes |
| `classify` | Classification labels | Top class and optional confidence |
| `semantic` | Class map overlay | Classes and valid pixel coverage |
| `depth` | Transparent yellow to pink overlay | Valid pixels and raw value range |
| `count` | Current visible object count | Objects, classes and visible tracks |
| `heatmap` | Smooth accumulated density overlay | Objects, samples and vertical activity legend |

Omit `demo` to render the result's available fields automatically. Inference task
selection requires compatible weights. Count and heatmap work with detect,
segment, pose or OBB weights. `task="segment", tracking=True` enables tracked
segmentation; `demo="track"` enables tracking on a compatible checkpoint.

Counts describe objects in the current filtered frame. Heatmap samples include
repeated observations across frames. Neither means unique lifetime objects or
line crossings. A new source, frame size or `visualizer.reset()` clears history.

## Summary panel

The rounded panel appears in the top right by default. Title and subtitle are
optional strings passed directly to `Visualizer`:

```python
visualizer = Visualizer(
    summary_title="Production overview",
    summary_subtitle="Ampoule counting",
    summary_items={"Line": "A", "Status": "Running"},
)
result = model("image.jpg")[0]
frame = visualizer.annotate(result)
```

Use `summary=False` to hide the panel. The title defaults to the task name;
the subtitle is omitted when `None`. Text is shortened to fit the panel, while
full metric values remain accessible through `visualizer.last_summary.items`.

Custom metrics can also update per frame:

```python
visualizer = Visualizer(
    summary_title="Vehicles",
    summary_subtitle="North entrance",
    summary_items=lambda state: {"Cars": state.class_counts.get("car", 0)},
)
```

The callback receives `FrameSummary` with `task`, sequence `frame`, default
`items`, `class_counts` and current `track_ids`. Its returned mapping adds or
overrides cards. Static mappings are copied during construction. Raw BGR frames
show image dimensions and custom cards, since they have no predictions.

| Argument | Default | Meaning |
| --- | --- | --- |
| `summary` | `True` | Show the panel |
| `summary_title` | `None` | Custom title; otherwise the task name |
| `summary_subtitle` | `None` | Optional second line below the title |
| `summary_items` | `None` | Metric mapping or callback |
| `summary_width` | Automatic | Panel width in pixels, clamped to the frame |
| `summary_columns` | `2` | Maximum card columns, from 1 through 3 |
| `summary_margin` | `16` | Inset from the top and right edges |
| `summary_opacity` | `0.92` | Dark blue background opacity |

## Heatmaps and depth

```python
visualizer = Visualizer(
    "input.mp4",
    model=model,
    demo="heatmap",
    summary_title="Traffic activity",
    summary_subtitle="Accumulated object observations",
    heatmap_opacity=0.72,
    heatmap_radius=48,
    heatmap_contours=True,
    show=True,
)
visualizer()
```

Heatmaps preserve the scene with transparent density shading. The summary's
activity scale is **vertical**, with high activity at the top and low activity
at the bottom. The overlay itself follows the frame in either orientation;
it is not rotated or stretched to a fixed aspect ratio.

Density is normalized to the current peak accumulated activity. The automatic
Gaussian spread follows 4.5% of the frame's shorter side. Heatmap mode hides
boxes, labels, masks, pose and trails by default when using direct arguments;
explicit layer arguments or a supplied Style override those defaults.

| Argument | Default | Meaning |
| --- | --- | --- |
| `heatmap_opacity` | `0.72` | Maximum density overlay opacity |
| `heatmap_radius` | Automatic | Gaussian standard deviation in image pixels |
| `heatmap_contours` | `True` | Subtle relative density contours |
| `depth_opacity` | `0.45` | Blend depth colors with the original scene |
| `mask_opacity` | `0.45` | Instance and semantic mask opacity |

Depth is normalized per frame: lower values are yellow, higher values pink.
These are model values, not RGB brightness. Nonfinite pixels remain unchanged;
the visualization does not assume metric units. Opacity `0` hides an overlay;
`1` makes it solid. Depth, heatmap and mask opacity settings are independent.

## Labels and geometry

Automatic sizing uses `scale = min(height, width) / 1080`:

- Outline width: `max(1, round(4 * scale))` pixels, thinner on small boxes.
- Arial font size: `max(10, round(30 * scale))` pixels, independent of box size.

| Frame size, including portrait equivalents | Outline | Label font |
| --- | --- | --- |
| 1280 × 720 | 3 px | 20 px |
| 1920 × 1080 | 4 px | 30 px |
| 3840 × 2160 | 8 px | 60 px |
| 7680 × 4320 | 16 px | 120 px |

Explicit sizes remain fixed pixel overrides. Labels touch box outlines by
default and retain padding inside the rounded background. Tracking labels show
only numeric IDs. Trails retain at most ten actual frames.

| Argument | Default | Meaning |
| --- | --- | --- |
| `line_width`, `font_size` | `None` | Automatic sizing, or fixed positive pixels |
| `corner_radius`, `label_radius` | `8`, `6` | Box and label corner radii |
| `label_padding`, `label_margin` | `6`, `0` | Internal padding and gap from the box |
| `show_conf`, `track_ids` | `True` | Confidence text and tracked object labels |
| `text_label_color` | Automatic | Text contrast, or an RGB tuple |
| `font_path` | Automatic | Licensed Arial font path |
| `boxes`, `labels`, `masks`, `pose`, `trails` | `True` | Object layers; heatmap defaults differ |
| `min_confidence`, `classes` | `0.0`, `None` | Rendering filters |
| `trail_length`, `trail_max_gap` | `10`, `10` | Trail frame limit and missing track expiry |
| `keypoint_radius` | Automatic | Keypoint size in pixels |
| `keypoint_confidence` | `0.25` | Keypoint visibility threshold |
| `skeleton` | Automatic | COCO edges for 17 points; custom index pairs otherwise |
| `ignored_semantic_ids` | `(255, -1)` | Semantic IDs excluded from overlays |

Colors and annotation typography follow the
[Ultralytics brand guidelines](https://www.ultralytics.com/brand). Arial must be
installed or supplied through `font_path` or `ULTRAVIZ_FONT`; it is not bundled.
Use `labels=False, summary=False` to render without loading a font. All color
arguments use RGB; input and output image arrays use OpenCV BGR.

## Save, inspect and time results

```python
visualizer = Visualizer(
    "input.mp4",
    model=model,
    save="output/annotated.mp4",
    fps=30,
    timing=True,
)
operation = visualizer()
print(operation.frames, operation.output)
```

`save` accepts one image path, an MP4/AVI video path, or a directory for numbered
PNG frames. Set the source FPS explicitly for video exports. Output is silent
video, with constant even frame dimensions. Existing files are protected unless
`overwrite=True`; source paths remain protected. Output directories are created
automatically. Video files are published only after successful processing.

`timing=True` logs inference, visualization and their sum in seconds to the
console. Timings never appear on frames. Inference uses the result's metadata;
visualization includes conversion and rendering. Total excludes decoding,
preprocessing, postprocessing, preview waits and encoding. Unknown values show
`n/a`. Access `last_timing`, or pass `on_timing=callback` for per frame records.

To consume predictions or annotated frames yourself:

```python
from ultraviz import process

source = Visualizer("input.mp4", model=model)
for frame in process(source):
    raw_result = source.last_result
    # Use frame and raw_result here.
```

Iterating `source` directly yields raw Results; `on_result=callback` receives
each prediction during normal processing. Close an iterator when stopping early.
Sources are single pass. Reuse the loaded model for a new source; one Visualizer
owns one sequential stream and is not thread safe.

Source inference settings go in `inference={...}`, for example
`inference={"imgsz": 640, "device": "cpu"}`. Defaults stream one frame at a time
without buffering. Dictionaries, argparse Namespace and Style configurations
are also accepted; direct keywords take precedence. Unknown options fail early.

Other backends can supply `Prediction(image, instances=...)` with `Instance`
objects, or `Prediction(semantic=..., image=...)` and `Prediction(depth=...,
image=...)`. Masks and dense maps must match the image dimensions. Pose points
use `(K, 2)` or `(K, 3)` arrays; OBB corners use `(4, 2)` coordinates.
Input frames are never modified.

## CLI

```bash
ultraviz --model yolo26n.pt --source image.jpg --output annotated.png
ultraviz --model yolo26n.pt --source input.mp4 --output activity.mp4 --fps 30 \
  --demo heatmap --summary-title "Traffic activity" \
  --summary-subtitle "Camera 01" --timing
ultraviz --help
```

Use `--no-summary` to hide the panel, `--hide-conf` for confidence text,
`--heatmap-radius` and `--heatmap-opacity` to adjust density, and
`--no-heatmap-contours` to hide contour accents.

## Development

```bash
pip install -e '.[dev,ultralytics]'
ruff check .
ruff format --check .
mypy
pytest
python -m build
```

Tests use synthetic predictions and real Ultralytics Results without downloading
models or opening GUI windows. Arial tests skip when the font is unavailable;
Ultralytics integration tests skip when the optional backend is absent.

## License

[AGPL-3.0-only](LICENSE). This independent library does not imply Ultralytics
endorsement. Ultralytics and Arial retain their own licenses and trademarks.
