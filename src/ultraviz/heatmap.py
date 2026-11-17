"""Smooth, transparent density shading using Ultralytics brand colors."""

from functools import lru_cache

import cv2
import numpy as np

from .branding import DARK_BLUE, NEON_PINK, NEON_YELLOW
from .predictions import Array, Image


@lru_cache(maxsize=1)
def heatmap_lut() -> Image:
    """Return a shared, immutable BGR ramp for the density field and its legend."""
    stops = np.array(
        [DARK_BLUE, (4, 42, 255), (0, 255, 255), (118, 255, 214), NEON_YELLOW, NEON_PINK]
    )
    positions = (0, 0.18, 0.40, 0.62, 0.82, 1)
    ramp = np.column_stack(
        [np.interp(np.linspace(0, 1, 256), positions, stops[:, channel]) for channel in range(3)]
    )
    bgr = np.ascontiguousarray(np.rint(ramp[:, ::-1]).astype(np.uint8)[:, None, :])
    bgr.flags.writeable = False
    return bgr


def blend_heatmap(
    image: Image,
    density: Array,
    *,
    radius: float | None,
    opacity: float,
    contours: bool,
) -> None:
    """Blend a smooth density field locally, preserving every inactive scene pixel.

    Density uses an eight pixel grid. Radius is the Gaussian standard deviation
    in image pixels; its default follows the image short edge in either orientation.
    Values show relative accumulated activity, not calibrated probabilities.
    """
    if opacity == 0 or not density.any():
        return
    height, width = image.shape[:2]
    sigma = max(0.5, (radius if radius is not None else min(height, width) * 0.045) / 8)
    smooth = cv2.GaussianBlur(
        density.astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_CONSTANT
    )
    intensity = smooth / max(float(smooth.max()), 1e-12)
    active = (intensity > 0.035).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(active)
    if w == 0 or h == 0:
        return
    # Include interpolation neighbors so the ROI edge fades into the source.
    x0, y0 = max(0, x - 1), max(0, y - 1)
    x1, y1 = min(density.shape[1], x + w + 1), min(density.shape[0], y + h + 1)
    field = intensity[y0:y1, x0:x1]
    px, py = x0 * 8, y0 * 8
    scaled_width, scaled_height = (x1 - x0) * 8, (y1 - y0) * 8
    levels = np.rint(field * 255).astype(np.uint8)
    tint = cv2.applyColorMap(levels, heatmap_lut())
    alpha = np.clip((field - 0.035) / 0.965, 0, 1) ** 0.7 * opacity
    # Shade on the small grid, then upscale premultiplied colors. OpenCV's
    # integer blend avoids allocating several full resolution float images.
    premultiplied = np.rint(tint * alpha[:, :, None]).astype(np.uint8)
    scaled = cv2.resize(premultiplied, (scaled_width, scaled_height))[: height - py, : width - px]
    weights = cv2.resize(np.rint(alpha * 255).astype(np.uint8), (scaled_width, scaled_height))[
        : height - py, : width - px
    ]
    roi = image[py : py + scaled.shape[0], px : px + scaled.shape[1]]
    inverse = cv2.cvtColor(255 - weights, cv2.COLOR_GRAY2BGR)
    cv2.multiply(roi, inverse, scale=1 / 255, dst=roi)
    cv2.add(roi, scaled, dst=roi)
    if contours:
        field = cv2.resize(field, (scaled_width, scaled_height))[: height - py, : width - px]
        # Quiet contour accents mark levels without hiding object texture.
        accents = roi.copy()
        for level in (0.25, 0.55, 0.85):
            paths, _ = cv2.findContours(
                (field >= level).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                accents,
                paths,
                -1,
                (255, 255, 255),
                max(1, round(min(height, width) / 1080)),
                cv2.LINE_AA,
            )
        cv2.addWeighted(accents, 0.2 * opacity, roi, 1 - 0.2 * opacity, 0, dst=roi)
