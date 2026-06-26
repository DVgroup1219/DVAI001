"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_jpeg(tmp_path: Path) -> Path:
    """Create a synthetic JPEG for testing."""
    rgb = np.zeros((200, 300, 3), dtype=np.uint8)
    rgb[:, :, 0] = 120
    rgb[:, :, 1] = 160
    rgb[:, :, 2] = 90

    path = tmp_path / "IMG001.JPG"
    img = Image.fromarray(rgb, mode="RGB")
    img.save(path, format="JPEG", quality=90)
    return path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
