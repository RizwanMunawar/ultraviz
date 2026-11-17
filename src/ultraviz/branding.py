"""Ultralytics annotation colors. Public colors use RGB; OpenCV uses BGR.

Source: https://www.ultralytics.com/brand (bounding boxes palette).
"""

from functools import lru_cache

RGB = tuple[int, int, int]
PALETTE: tuple[RGB, ...] = (
    (4, 42, 255),
    (11, 219, 235),
    (243, 243, 243),
    (0, 223, 183),
    (17, 31, 104),
    (255, 111, 221),
    (255, 68, 79),
    (204, 237, 0),
    (0, 243, 68),
    (189, 0, 255),
    (0, 180, 255),
    (221, 0, 186),
    (0, 255, 255),
    (38, 192, 0),
    (1, 255, 179),
    (125, 36, 255),
    (123, 0, 104),
    (255, 27, 108),
    (252, 109, 47),
    (162, 255, 11),
)
DARK_BLUE: RGB = (17, 31, 104)
WHITE: RGB = (255, 255, 255)
NEON_YELLOW: RGB = (225, 255, 37)
NEON_PINK: RGB = (255, 100, 218)


@lru_cache(maxsize=256)
def color(class_id: int) -> RGB:
    """Return a deterministic class color, cycling after twenty classes."""
    return PALETTE[class_id % len(PALETTE)]


@lru_cache(maxsize=20)
def text_color(background: RGB) -> RGB:
    """Choose dark blue or white for the higher WCAG text contrast."""

    def luminance(rgb: RGB) -> float:
        """Calculate relative luminance from linearized RGB channel values."""
        values = [v / 255 for v in rgb]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    value = luminance(background)
    dark = luminance(DARK_BLUE)
    return WHITE if (1.05 / (value + 0.05)) > ((value + 0.05) / (dark + 0.05)) else DARK_BLUE
