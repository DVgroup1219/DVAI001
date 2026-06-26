"""Shared fixtures for Module 03 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_rgb() -> np.ndarray:
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[:, :, 1] = 140
    rgb[:80, :, 2] = 200
    return rgb


@pytest.fixture
def green_rgb() -> np.ndarray:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    rgb[:, :, 1] = 180
    return rgb


@pytest.fixture
def sample_jpeg(tmp_path: Path) -> Path:
    rgb = np.zeros((200, 300, 3), dtype=np.uint8)
    rgb[:, :, 1] = 140
    path = tmp_path / "IMG001.JPG"
    Image.fromarray(rgb, mode="RGB").save(path, format="JPEG")
    return path
