# ruff: noqa: D100, D103
# Scenario names document tests; library definitions require docstrings.
import numpy as np
import pytest

from ultraviz import Style
from ultraviz.style import load_font


@pytest.fixture
def image():
    return np.zeros((96, 128, 3), dtype=np.uint8)


@pytest.fixture
def style():
    return Style(labels=False, summary=False)


@pytest.fixture
def arial():
    try:
        return load_font(Style(summary=False))
    except FileNotFoundError:
        pytest.skip("Licensed Arial is not installed")
