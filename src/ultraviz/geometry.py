"""Construct smooth rounded outlines while preserving rotated box geometry."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .predictions import Array

_T = np.linspace(0, 1, 9)[None, :, None]
_START = (1 - _T) ** 2
_CORNER = 2 * (1 - _T) * _T
_END = _T**2


def rounded_polygon(points: Array, radius: float) -> Array:
    """Return a closed polygon path with tangent quadratic curves at its corners.

    Distances are bounded by adjacent edge lengths so tiny or thin boxes do
    not fold over. A zero radius preserves the original polygon exactly.
    """
    vertices = np.asarray(points, dtype=np.float64)
    if radius <= 0:
        return vertices
    previous = np.roll(vertices, 1, axis=0) - vertices
    following = np.roll(vertices, -1, axis=0) - vertices
    before = np.linalg.norm(previous, axis=1)
    after = np.linalg.norm(following, axis=1)
    distance = np.minimum(radius, np.minimum(before, after) / 2)
    start = vertices + previous * (distance / np.maximum(before, 1e-12))[:, None]
    end = vertices + following * (distance / np.maximum(after, 1e-12))[:, None]
    curves = _START * start[:, None] + _CORNER * vertices[:, None] + _END * end[:, None]
    return np.asarray(curves.reshape(-1, 2))


@lru_cache(maxsize=512)
def rounded_rectangle(width: int, height: int, radius: int) -> Array:
    """Cache a read only local outline; callers translate it without modifying it."""
    path = rounded_polygon(np.array([[0, 0], [width, 0], [width, height], [0, height]]), radius)
    path.flags.writeable = False
    return path
