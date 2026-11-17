"""Headless inference and export entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .inference import DEMOS, TASKS, predict, resolve_demo
from .processing import ProcessSummary, process


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run inference and explicitly export one image or a video."""
    parser = argparse.ArgumentParser(description="Ultraviz: branded computer vision visualization")
    parser.add_argument("--model", required=True, help="Ultralytics weights path or model name")
    parser.add_argument(
        "--source", required=True, help="Input image/video path, URL or camera index"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--demo", choices=DEMOS)
    parser.add_argument("--hide-conf", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Show task metrics in the top right")
    parser.add_argument("--summary-title")
    parser.add_argument("--summary-width", type=int)
    parser.add_argument("--summary-columns", type=int, default=2)
    parser.add_argument("--summary-margin", type=int, default=16)
    parser.add_argument("--summary-opacity", type=float, default=0.92)
    parser.add_argument("--depth-opacity", type=float, default=0.45)
    parser.add_argument(
        "--timing",
        nargs="?",
        const="console",
        default=False,
        choices=("console",),
        help="Log per frame durations to the console only",
    )
    parser.add_argument(
        "--task", choices=TASKS, help="Model task; inferred from weights by default"
    )
    parser.add_argument("--track", action="store_true", help="Use model.track()")
    parser.add_argument("--fps", type=float, help="Required for .mp4/.avi video output")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device")
    parser.add_argument("--font", type=Path, help="Licensed Arial.ttf path")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    video = args.output.suffix.lower() in {".mp4", ".avi"}
    if video and args.fps is None:
        parser.error("--fps is required for video output")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if args.output.resolve() == Path(args.source).resolve():
        parser.error("output must not overwrite the input source")
    results = None
    try:
        task, tracking = resolve_demo(args.demo, args.task, args.track)
        source = int(args.source) if args.source.isdecimal() else args.source
        results = predict(
            args.model,
            source,
            task=task,
            tracking=tracking,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        summary = cast(
            ProcessSummary,
            process(
                results,
                save=args.output,
                fps=args.fps if args.fps is not None else 30,
                overwrite=args.overwrite,
                labels=not args.no_labels,
                show_conf=not args.hide_conf,
                timing=args.timing,
                summary=args.summary,
                summary_title=args.summary_title,
                summary_width=args.summary_width,
                summary_columns=args.summary_columns,
                summary_margin=args.summary_margin,
                summary_opacity=args.summary_opacity,
                depth_opacity=args.depth_opacity,
                demo=args.demo,
                font_path=args.font,
            ),
        )
        print(f"Saved {summary.frames} frame(s) to {summary.output}")
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"ultraviz: {exc}\n")
    finally:
        close = getattr(results, "close", None)
        if callable(close):
            close()
    return 0
