"""Manage a single OpenCV preview window and its lifecycle."""

from __future__ import annotations

import os
import sys
from time import perf_counter

import cv2

from .predictions import Image


class PreviewWindow:
    """Own just our preview window and close it on completion or failure."""

    def __init__(self, enabled: bool, fps: float) -> None:
        """Configure playback timing and initialize ownership of the preview window."""
        self.enabled = enabled
        self.period = 1 / fps
        self._last_display: float | None = None
        self.opened = False
        self.stopped = False
        self.name = f"Ultraviz {id(self):x}"

    def __call__(self, image: Image) -> bool:
        """Display one frame and report whether processing should continue."""
        if not self.enabled:
            return True
        if not self.opened:
            if sys.platform.startswith("linux") and not (
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            ):
                raise RuntimeError("show=True requires a graphical desktop; use show=False here")
            try:
                cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            except cv2.error as exc:
                raise RuntimeError("OpenCV GUI support is required for show=True") from exc
            self.opened = True
        cv2.imshow(self.name, image)
        now = perf_counter()
        remaining = 0.0 if self._last_display is None else self._last_display + self.period - now
        key = cv2.waitKey(max(1, round(remaining * 1000))) & 0xFF
        self._last_display = perf_counter()
        self.stopped = (
            key in (27, ord("q")) or cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1
        )
        return not self.stopped

    def finish(self) -> None:
        # Keep the final frame visible, including streamed single images.
        """Wait for a key so the final frame remains visible."""
        if self.opened and not self.stopped:
            cv2.waitKey(0)

    def close(self) -> None:
        """Destroy only this preview window, tolerating a window already closed by the user."""
        if self.opened:
            try:
                cv2.destroyWindow(self.name)
            except cv2.error:
                pass  # The user may already have closed our window.
