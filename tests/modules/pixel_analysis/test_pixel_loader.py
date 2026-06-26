"""Tests for JPEG loader."""

from pathlib import Path

import pytest

from src.modules.pixel_analysis.exceptions import ImageLoadError
from src.modules.pixel_analysis.loader import JPEGLoader


def test_load_jpeg(sample_jpeg: Path) -> None:
    loaded = JPEGLoader().load(sample_jpeg)
    assert loaded.width == 300
    assert loaded.height == 200


def test_reject_non_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "file.png"
    path.write_bytes(b"x")
    with pytest.raises(ImageLoadError):
        JPEGLoader().load(path)
