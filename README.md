# Ultraviz

Visualize Ultralytics YOLO26 predictions for detection, instance segmentation,
pose estimation, oriented bounding boxes, semantic segmentation and tracking.
Classification and depth maps use the same annotation pipeline.

## Source inference and live display

```python
from ultralytics import YOLO
from ultraviz import Visualizer

visualizer = Visualizer(
    "1.mp4",
    model=YOLO("Newones/models/yolo26n-visdrone.pt"),
    demo="detect",
    show=True,
    show_conf=False,
)
summary = visualizer()
```

Passing `model=YOLO(...)` directly is supported. The loaded object is reused
for the full stream, with no second model load. This workflow also accepts an image path. The model must be supplied explicitly;
Ultraviz does not inspect a separate `model` variable in the caller's scope.
A weights path can be passed as `model="path/to/weights.pt"` instead of a loaded
YOLO object. Existing weights are read without modification.

Creating the Visualizer runs no inference. Predictions are requested one at a
time when the stream is consumed. `process(..., show=True)` displays each frame
before asking for the next prediction. Source mode defaults to `stream=True`,
`batch=1`, `stream_buffer=False` and `verbose=False`. No video file frame skipping is enabled by default. Changing batch size or inference settings remains explicit.

You can also configure display/export on the instance and run it directly:

```python
visualizer = Visualizer(
    "1.mp4",
    model=model,
    show=True,
    save="output/annotated.mp4",
    fps=30,
    mask_opacity=0.5,
    corner_radius=8,
    label_radius=6,
)
summary = visualizer()
```

## Existing YOLO results

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n.pt")
results = model("path/to/video.mp4", stream=True)
args = {"show": True, "line_width": 2}
visualizer = Visualizer(args)
summary = visualizer(results)
```

Use `stream=True` for live display. Calling `model(video)` without streaming
finishes inference for the entire video before returning its result list;
Ultraviz cannot display predictions before the model returns them.

Configuration accepts a dictionary, `argparse.Namespace`, `Style` object or
keywords. Keywords override dictionary/namespace values without modifying the
input. There is one public Visualizer class in `visualizer.py`.

## Extract predictions and frames

Source instances are single pass iterators of raw Ultralytics Results:

```python
results = Visualizer("1.mp4", model=model)
for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    obb = result.obb
```

To inspect predictions while displaying, pass `on_result=your_callback` to the
source constructor. The callback receives every raw Results object immediately
after inference. `results.last_result` exposes the most recent raw prediction
without storing the whole video. Use `contextlib.closing(results)` when breaking
out of raw iteration early. A consumed or closed source does not restart itself;
create a new source instance to run inference again with the same loaded model.

To extract annotated frames instead:

```python
results = Visualizer("1.mp4", model=model)
for frame in process(results):
    # Use each annotated BGR array here.
    pass
```

The source's style is inherited by `process`, and explicit style options can
override it. `show` and `save` on `process` control that operation; the source's
show/save settings apply when you call `results()` directly.
Already annotated BGR arrays or frame iterators can also be passed to `process`
for display or export, without running inference or drawing annotations again.

## Install

Requires Python 3.10 or newer:

```bash
pip install -e '.[ultralytics]'
```

For rendering without inference, use `pip install -e .`; the base dependencies
are NumPy, OpenCV and Pillow. Ultralytics, PyTorch, model loading and weight
files remain optional. This independent library does not imply Ultralytics
endorsement. The distribution, import and console command are named `ultraviz`.

## Options

```python
visualizer = Visualizer(
    show=False,
    save="output/annotated.mp4",
    fps=30,
    line_width=3,
    font_size=None,  # Automatic image based label sizing.
    mask_opacity=0.5,
    min_confidence=0.3,
    classes=frozenset({0, 2}),
    trails=True,
)
summary = visualizer(results)
print(summary.frames, summary.output, summary.stopped)
```

| Option | Default | Behavior |
| --- | --- | --- |
| `show` | `False` | Preview results in a graphical desktop window |
| `save` | `None` | Image path for one result; MP4/AVI path for video; directory for PNG frames |
| `fps` | `30` | Video export and preview frame rate |
| `overwrite` | `False` | Allow replacing an existing output |
| `line_width` | `None` | Automatic width based on image and object size; an integer overrides it |
| `font_size` | `None` | Automatic Arial size based on image and box dimensions; an integer overrides it |
| `corner_radius` | `8` | Rounded regular/rotated box corners, clamped for tiny boxes |
| `label_radius` | `6` | Rounded label background corners |
| `label_padding` | `6` | Pixels of space on each side of label text |
| `label_margin` | `0` | Labels touch boxes by default; a positive value adds a gap |
| `font_path` | Automatic | Location of a licensed Arial font |
| `mask_opacity` | `0.45` | Instance, semantic and heatmap overlay opacity |
| `boxes`, `labels`, `masks`, `pose`, `trails` | `True` | Enable or disable annotation layers |
| `show_conf` | `True` | Print confidence scores in untracked and classification labels |
| `text_label_color` | `None` | Automatic contrast, or an explicit RGB tuple such as `(255, 255, 255)` |
| `timing` | `False` | `True` or `"console"` logs per frame durations; never drawn on frames |
| `depth_opacity` | `0.45` | Blend the yellow to pink depth map with the original scene |
| `summary` | `False` | Enable the dynamic summary panel in the top right |
| `summary_title` | Automatic | Override the task heading |
| `summary_items` | `None` | Mapping or callback adding/overriding metric cards |
| `summary_width` | Automatic | Panel width in pixels, clamped to the image |
| `summary_columns` | `2` | Metric columns, from 1 through 3 |
| `summary_margin` | `16` | Inset from the top and right frame edges |
| `summary_opacity` | `0.92` | Dark blue panel background opacity; cards remain opaque |
| `demo` | `None` | Automatic mode, or a model task, count or heatmap |
| `track_ids` | `True` | Show only numeric IDs on tracked instances; False hides their labels |
| `min_confidence` | `0.0` | Filter instances by confidence |
| `classes` | `None` | Restrict instance and semantic class IDs |
| `trail_length` | `10` | Maximum positions and frame age; allowed range is 1 through 10 |
| `trail_max_gap` | `10` | Frames before a missing track record expires |
| `keypoint_confidence` | `0.25` | Threshold for keypoint visibility |
| `keypoint_radius` | `None` | Follow the resolved outline width; an integer overrides it |
| `skeleton` | Automatic | COCO edges for 17 points; custom zero based edge pairs otherwise |
| `ignored_semantic_ids` | `(255, -1)` | Semantic IDs left uncolored |

Automatic sizing uses `scale = min(height, width) / 1080`. The default outline
is `max(1, round(4 * scale))` pixels and Arial text is
`max(10, round(30 * scale))` pixels. Using the shorter side gives the same
sizes for portrait and landscape frames. Large images continue scaling
without a 4K cap; small objects can receive thinner outlines and smaller labels.
Explicit `line_width` and `font_size` values remain fixed pixel overrides.

| Frame size (landscape or portrait) | Outline | Arial size |
| --- | --- | --- |
| 1280 × 720 / 720 × 1280 | 3 px | 20 px |
| 1920 × 1080 / 1080 × 1920 | 4 px | 30 px |
| 2560 × 1440 / 1440 × 2560 | 5 px | 40 px |
| 3840 × 2160 / 2160 × 3840 | 8 px | 60 px |
| 7680 × 4320 / 4320 × 7680 | 16 px | 120 px |

These sizes use the image's pixel resolution, including for exports intended
for print. Print DPI metadata does not change the rendered pixel dimensions.

Unknown options raise `TypeError`. A `Style` object can also be passed as
`args` or `style=...`; direct style options override its fields.

With `show` or `save`, the call processes the complete input and returns a
`ProcessSummary` without retaining video frames. Q/Escape stops the preview;
any key closes the final frame. Early stopping saves frames processed so far.
The preview requires an OpenCV GUI and graphical desktop. Exported video has
no audio. MP4 uses `mp4v`; AVI uses `MJPG`; video dimensions must be constant
and even. FPS defaults to 30, so supply your source FPS when exporting.

Image suffixes require exactly one result. A directory without a suffix saves
numbered `frame_000001.png` images. Output directories are created as needed.
Existing files are protected unless `overwrite=True`; known result source
paths remain protected. Video output is published only after successful
processing. A directory export retains completed images if a later frame fails.
Publishing without overwrite requires a filesystem supporting hard links.

## Flat style arguments and frame timing

No Style object is needed. Pass style fields directly to Visualizer or process:

```python
from ultralytics import YOLO
from ultraviz import Visualizer

visualizer = Visualizer(
    "1.mp4",
    model=YOLO("Newones/models/yolo26n-visdrone.pt"),
    font_size=36,
    label_padding=12,
    show_conf=False,
    text_label_color=(255, 255, 255),  # RGB, not OpenCV BGR.
    timing=True,
    show=True,
)
summary = visualizer()
```

`timing=True` or `timing="console"` prints a summary immediately after each
frame in console logs only. Timings are never drawn on the frame or saved
into the video. Timing remains off by default.

```text
Frame 1 | inference: 0.031000 s | visualization: 0.006000 s | total: 0.037000 s
```

The numbers above are an illustrative example. Inference uses Ultralytics'
reported per image inference duration, converted from milliseconds to seconds.
Visualization measures result conversion, copying and annotation. Total is their
sum; it excludes decoding, model preprocessing/postprocessing, preview waits,
encoding, logging and callbacks. It is not complete
pipeline wall time. Missing inference metadata displays `n/a`, including total,
rather than a fabricated zero. Raw BGR frames have no inference metadata.
[Ultralytics Results speed metadata](https://docs.ultralytics.com/modes/predict/#working-with-results).

`visualizer.last_timing` exposes the latest immutable FrameTiming record with
`inference_seconds`, `visualization_seconds` and `total_seconds`. Optional
`on_timing=your_callback` receives a record for every annotated frame, even with
console logging disabled. No record history is retained internally. Frame
numbers count annotations across the instance lifetime; resetting annotation
history does not reset those numbers. Native Prediction inputs can supply
`inference_seconds=...` when using another model backend. Console timing does
not require Arial.

`text_label_color=None` preserves automatic white/dark blue contrast. An RGB
override applies consistently to object labels and count captions. Summary cards retain automatic contrast.
Class colors remain in the Ultralytics palette. Style objects are still accepted
for reuse, but every field can be supplied directly and flat options take
precedence over fields in a supplied Style.

## Dynamic task summary

Enable the rounded summary panel with `summary=True`. All settings are direct
Visualizer arguments. The panel uses the Ultralytics dark blue background,
neon yellow, mint, bright blue and pink cards, rounded corners, and Arial.
It scales with the frame and stays in the top right. Long titles and values
are shortened to fit; complete values remain available in `last_summary`.

```python
visualizer = Visualizer(
    "1.mp4",
    model=YOLO("Newones/models/yolo26n-visdrone.pt"),
    demo="count",  # Also works with heatmap, segment, pose, obb, track, etc.
    summary=True,
    summary_title="Vehicles summary",
    summary_items={"Shift": "Morning"},
    summary_columns=2,
    depth_opacity=0.4,
    timing=True,  # Console only.
    show=True,
)
visualizer()
print(visualizer.last_summary.items)
```

Default cards update on every frame using filtered predictions:

| Task | Default metrics |
| --- | --- |
| Raw BGR frame | Image width and height |
| Detect / count | Objects in frame, visible classes, tracks when available |
| Tracking | Objects in frame, classes, visible tracks and IDs |
| Heatmap | Objects in frame, accumulated center samples since sequence reset |
| Segment | Instances, masks, union mask coverage |
| Pose | Poses, visible keypoints using the configured threshold |
| OBB | Oriented boxes, classes |
| Semantic | Classes and pixel coverage after ignored IDs and class filters |
| Classify | Top class and confidence, respecting `show_conf=False` |
| Depth | Finite pixel coverage and raw minimum/maximum values |

Count means current visible objects. Heatmap samples include repeated observations
across frames. Neither metric claims unique lifetime objects or line crossings.
Depth values have the units supplied by your model; the panel does not assume meters.

Use a callback for live custom metrics. It receives an immutable `FrameSummary`
with `task`, sequence `frame`, default `items`, `class_counts`, and current
`track_ids`. Return a mapping to add cards or override default names:

```python
visualizer = Visualizer(
    summary=True,
    summary_items=lambda state: {"Cars": state.class_counts.get("car", 0)},
)
frame = visualizer.annotate(results[0])
```

Static mappings are copied at construction. Callbacks run once per annotated
prediction frame, including empty results, and can read your own application
metrics such as cycle status. Raw BGR frames show image dimensions and custom
cards, since they have no prediction metadata. Summary calculations and font loading are skipped while
`summary=False`. Processing times remain in console logs, outside this panel.

## Tracking labels and spacing

```python
visualizer = Visualizer(
    "1.mp4",
    model=model,
    tracking=True,
    show=True,
    label_padding=8,
    label_margin=0,
)
summary = visualizer()
```

An instance with a track ID displays only that numeric ID, for example `42`.
No class name, confidence or prefix is added. This also applies to tracked
segmentation, pose and OBB results. `track_ids=False` hides tracked labels.
Untracked detections keep their normal class/confidence labels.

`label_padding` controls space inside the rounded background around all four
sides of the text. `label_margin=0` attaches the label background to its box.
Positive values deliberately add separation and an image edge inset. A label prefers the area above its
anchor and falls below if needed. Tiny frames clamp the spacing to fit.
Classification rows use the same padding and margin settings. Every task's
text passes through this shared label renderer.

## Returned frames

Without `show` or `save`, `visualizer(results)` returns annotated BGR arrays:

```python
visualizer = Visualizer()
frames = visualizer(model("image.jpg"))  # A list, even for one image.
frame = visualizer(model("image.jpg")[0])  # A single image array.

results = model("video.mp4", stream=True)
for frame in visualizer(results):
    # Use each annotated frame here.
    pass
```

List and tuple inputs return lists. Iterator inputs stay lazy and must be
consumed. Use `contextlib.closing(visualizer(results))` if stopping early so the
upstream iterator is closed. Input images are borrowed and remain unchanged;
returned frames are copies. All images use `uint8` BGR arrays `(H, W, 3)`.

A `Visualizer` belongs to one sequential stream and is not thread safe. Lists
and iterators start a fresh sequence. Single result calls retain tracking
history for incremental frame processing. Source or image size changes reset
history; call `visualizer.reset()` when seeking or switching unrelated videos
with the same source path. The annotation engine and loaded font are reused.
`annotate()` and `stream()` offer drawing without preview or export.

## Demo modes and confidence text

Use `demo` to select a workflow. Model task demos require matching weights.
Count and heatmap use instance predictions from detection, segmentation, pose
or OBB weights; they are visualization modes, not separate model architectures.

| `demo` | Behavior |
| --- | --- |
| Omitted or `None` | Automatically render the model's results |
| `detect`, `segment`, `pose`, `obb` | Select and validate the corresponding model task |
| `semantic`, `depth`, `classify` | Select and validate a dense or classification model task |
| `track` | Enable tracking on compatible weights, showing numeric IDs only |
| `count` | Display the current visible instance count |
| `heatmap` | Accumulate detection center density for the current sequence |

```python
visualizer = Visualizer("1.mp4", model=model, demo="count", show_conf=False, show=True)
summary = visualizer()
print(visualizer.count)  # Visible objects in the most recently annotated frame.

visualizer = Visualizer("1.mp4", model=model, demo="heatmap", show=True)
summary = visualizer()
```

Counting respects class and confidence filters and reports the current frame,
not line crossings or unique lifetime identities. Heatmaps use a coarse spatial
grid and never retain the video frames. `reset()` clears density and trails.
`boxes=False` and `labels=False` give a heatmap without object overlays.

`show_conf=False` hides printed scores; it does not change model predictions or
filtering. Use `inference={"conf": 0.4}` to set a model threshold. Tracked object
labels always contain only their numeric ID. The old Visualizer/process keyword
`confidence` remains an alias for `show_conf`; Style uses `show_conf` directly.

The default trail keeps at most 10 positions from the last 10 actual frames.
Set `trail_length=5` for a shorter trail. Values above 10 are rejected, and
missing detections cannot stretch a trail's time span beyond that window.

## Selecting a task

The default is automatic: the model weights determine inference, and result
fields determine annotation. You can select and validate the task explicitly:

```python
visualizer = Visualizer("image.jpg", model="yolo26n-seg.pt", task="segment", show=True)
summary = visualizer()

visualizer = Visualizer("video.mp4", model=model, task="track", show=True)
summary = visualizer()
```

| `task` | Meaning |
| --- | --- |
| Omitted or `None` | Infer from the model |
| `detect` | Object detection |
| `segment` | Instance segmentation |
| `pose` | Pose estimation |
| `obb` | Oriented bounding boxes |
| `classify` | Classification |
| `semantic` | Semantic segmentation |
| `depth` | Depth estimation |
| `track` | Tracking using the model's compatible task |

Task selection requires matching weights. `task="segment"` cannot convert a
detection checkpoint into a segmentation model; mismatches raise before
prediction runs. Tracking supports detect, segment, pose and OBB weights.
For tracked segmentation use `task="segment", tracking=True`, or use
`task="track"` with a segmentation checkpoint. Task settings can also be supplied
in the configuration dictionary or Namespace. Inference model task checks run
when the lazy source is first consumed. Existing Results objects always render
according to their actual fields; task selection does not invent missing data.

The console also supports `--demo count`, `--demo heatmap`, `--hide-conf` and
`--summary`, `--summary-title "Vehicles summary"` and `--timing` for console logs.
Use `--task segment` or `--task track`; `--track` is retained
for enabling tracking with an explicitly selected model task.

## Supported tasks

| Task | Required results | Output |
| --- | --- | --- |
| Detection | `boxes` | Axis aligned boxes and labels |
| Instance segmentation | `boxes` and `masks` | Bitmap masks preserving holes, plus boxes |
| Pose estimation | `boxes` and `keypoints` | Keypoints and skeleton edges |
| OBB | `obb` | Four rotated corners and labels |
| Tracking | IDs from `model.track()` | IDs and bounded trails, including segment, pose and OBB tracks |
| Semantic segmentation | `semantic_mask` or `Prediction(semantic=...)` | Integer class map overlay |
| Classification | `probs` | Top five class probabilities |
| Depth | `depth` or `Prediction(depth=...)` | Yellow to pink depth map |

```python
results = model.track("input.mp4", stream=True)
visualizer = Visualizer(save="output/tracks.mp4", trails=True, fps=30)
summary = visualizer(results)
```

Use models supported by your installed Ultralytics version. Ultraviz ships no
weights and does not run a second inference pass. Custom pose layouts draw
points unless `skeleton=((0, 1), (1, 2))` is configured. Depth colors are
normalized per frame. Lower values are yellow and higher values pink; these
are depth values, not the brightness of the RGB scene. Valid pixels use an
transparent overlay controlled by `depth_opacity=0.45`, independent of
`mask_opacity`, so scene details remain visible. Use `depth_opacity=0` to hide
the overlay or `1` for a solid map. NaN and infinite pixels remain unchanged. The gradient
is a visualization, not a metric depth scale.

## Ultralytics branding

Source of truth: [Ultralytics brand guidelines](https://www.ultralytics.com/brand),
reviewed November 17, 2026.

Annotation labels use **Arial**, as specified in the bounding boxes section.
**Archivo** is the corporate font for editorial material. Arial is discovered
in standard macOS, Windows and Linux locations. Set `font_path` or the
`ULTRAVIZ_FONT` environment variable to your licensed Arial file when needed.
The font family is verified, so a renamed substitute cannot bypass the check.
Arial is not redistributed. The automatic size is 2.5 percent of the shorter
image dimension, clamped to 10 through 64 pixels, with a smaller size for
small boxes. Outline widths and keypoint radii also scale automatically.
Explicit `font_size`, `line_width` and `keypoint_radius` values take precedence.
Box outlines and label backgrounds are rounded by default as requested;
`corner_radius=0` or `label_radius=0` restores square geometry. Missing Arial raises a setup error when labels
are drawn; `labels=False` renders geometry without loading a font.

Class colors use the official twenty entry bounding box palette in its
published order. Colors repeat after twenty classes and remain stable across
track IDs. By default, labels choose white or dark blue text by relative luminance for
contrast. `text_label_color` provides an explicit RGB override. Depth uses exact neon yellow (#E1FF25) at its low end and neon pink
(#FF64DA) at its high end, with continuous interpolation between them. No logos are
altered or redistributed.

## Inference performance

Model execution settings belong to Ultralytics. Pass them to `model(...)`
directly, or use `inference={...}` in source mode:

```python
model = YOLO("yolo26n.pt")  # Load once and reuse.
results = model("video.mp4", stream=True, device=0, quantize=16)
visualizer = Visualizer(save="annotated.mp4", fps=30)
summary = visualizer(results)
```

For the source workflow, the equivalent is
`Visualizer("video.mp4", model=model, inference={"device": 0, "quantize": 16})`.

`device=0` and `quantize=16` are optional settings for a compatible GPU and model
backend; omit them for other hardware. Current Ultralytics uses `quantize=16`
for FP16; its older `half` flag is deprecated. Further improvements depend on the
actual device and accuracy requirements:

- Use `stream=True` for bounded result memory; it does not itself guarantee
  lower inference latency.
- Reuse the YOLO model and Visualizer. Font discovery happens once per engine.
- Benchmark `imgsz`, batch size and smaller models against your accuracy needs.
- Use FP16 where supported. Consider TensorRT for compatible NVIDIA GPUs,
  OpenVINO for suitable CPUs, or CoreML for supported Apple deployments.
- Keep `show=False` for throughput measurements; preview intentionally waits
  between frames. Measure preprocessing, inference, conversion, annotation and
  encoding separately after warmup.

The result adapter transfers box metadata to CPU once, instead of separately
transferring coordinates, classes, confidence and IDs. OBB corner geometry,
masks and keypoints still require their own transfers. Label contrast colors
and fonts are cached. Disabled mask and pose layers skip their tensor
transfers entirely, and fully transparent mask blending is skipped.
Repeated labels use bounded cached sprites composited into local image regions,
so labeling no longer converts the complete frame through Pillow.

A local CPU smoke test used the existing VisDrone weights and eight 1920 by 1080
frames from `1.mp4`, with `imgsz=640` and around 90 detections on the first frame.
Median annotation time across frames 2 through 8 fell from 49.38 ms before label
sprite caching to 9.01 ms afterward. Model inference medians were 30.74 ms and
30.43 ms respectively; this is a rendering improvement, not a measured model
speedup. The short sample does not predict performance on other devices or
videos. User weights were not modified. No CUDA or MPS device was available in
the test environment.

A longer followup with the updated label padding measured 55 frames after five
warmup frames, averaging 88.9 detections per 1080p frame. Median annotation was
7.84 ms, its 95th percentile was 9.12 ms, and median model inference was 31.52 ms.
The measured decode/inference/annotation throughput was 23.3 FPS without preview
or output encoding. These detection measurements do not establish throughput
for dense segmentation masks or every production workload. Tests also exercise
600 sequential track IDs to verify bounded label cache and trail history.

The code audit additionally measured 30 sparse 80 by 120 masks in a 1080p
frame. Replacing full frame boolean blending with bounded mask regions reduced
median annotation from 272.18 ms to 22.43 ms. Mask construction dropped from
10.63 ms to 5.74 ms after removing duplicate copies and scans. This is a
synthetic sparse mask case; dense/full frame masks have different costs.

Caching rounded rectangle paths reduced median detection annotation from
7.84 ms to 6.45 ms on the same 55 measured video frames after five warmup frames.
The new 95th percentile was 7.48 ms; model inference was 30.90 ms and measured
processing throughput was 24.5 FPS without preview or encoding. These timings
are local samples, not guarantees across models or devices.

References: [prediction settings](https://docs.ultralytics.com/modes/predict/)
and [export backends](https://docs.ultralytics.com/modes/export/).

## Other model backends

```python
import numpy as np
from ultraviz import Instance, Prediction, Visualizer

image = np.zeros((480, 640, 3), dtype=np.uint8)
prediction = Prediction(
    image,
    instances=(
        Instance(class_id=0, name="person", confidence=0.92, box=(100, 80, 250, 400), track_id=7),
    ),
)
frame = Visualizer()(prediction)
```

Optional instance fields are `corners` `(4, 2)`, boolean `mask` `(H, W)` and
`keypoints` `(K, 2)` or `(K, 3)` with confidence. Coordinates use original image
pixels. Dense maps must match the image `(H, W)`. Geometry and maps are copied
on construction. The Ultralytics adapter assumes standard centered letterbox
masks; custom preprocessing should restore masks before creating `Prediction`.

## Command line

```bash
ultraviz --model yolo26n.pt --source image.jpg --output annotated.png
ultraviz --model yolo26n-seg.pt --source input.mp4 --output annotated.mp4 --fps 30
ultraviz --help
```

The console implementation is `cli.py`. There is no `__main__.py`.
`process`, `save_image` and `save_video` remain available as convenience
or lower level functions, sharing the same annotation implementation. The
recommended user entry point is `Visualizer`.

## Audit cleanup

The CLI now uses the same processing/export implementation as the Python API.
Source processing reuses its existing annotation engine and font/label caches.
Video export no longer repeats filesystem checks for an unchanged source on
every frame, and directory export creates its directory once. Label sprites are
bounded by both 256 entries and an 8 MiB byte budget.

The redundant `render()` helper was removed; use `Visualizer()(result)` or
`process(result)`. Unused `Prediction.names` was removed; native instances carry
`Instance.name`, while raw Ultralytics Results retain their names dictionary.
No task implementations or requested rendering features were removed.

Additional checks reject invalid boolean settings, unsupported task/model
combinations, missing class IDs and nonnumeric geometry. Cleanup releases the
input stream even when preview teardown fails. Depth normalization avoids
overflow for extreme finite values. Array ownership and rendered mask pixels
remain covered by regression tests.

## Development

```bash
pip install -e '.[dev,ultralytics]'
ruff check .
ruff format --check .
mypy
pytest
python -m build
```

Docstring linting covers library classes and public methods. Tests have scoped
`noqa` comments because descriptive test names document their scenarios.
Tests use synthetic data and real Ultralytics Results without model downloads
or GUI windows. Arial tests skip when unavailable; integration tests skip when
Ultralytics or PyTorch is absent. CI covers Python 3.10 through 3.13 on Linux,
macOS and Windows, with a separate integration job.

Modules have descriptive responsibilities: `visualizer.py`, `annotation.py`,
`predictions.py`, `adaptor.py`, `branding.py`, `style.py`,
`processing.py`, `preview.py`, `summary.py`, `timing.py`, `export.py`, `inference.py` and `cli.py`.

## License

[AGPL-3.0-only](LICENSE). Ultralytics and Arial retain their licenses and trademarks.
