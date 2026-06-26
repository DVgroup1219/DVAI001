"""Tests for JPEG loader."""

from pathlib import Path

import pytest

from src.modules.face_detection.loader import JPEGLoader, ImageLoadError


def test_load_jpeg(sample_jpeg: Path) -> None:
    loader = JPEGLoader()
    loaded = loader.load(sample_jpeg)
    assert loaded.width == 300
    assert loaded.height == 200


def test_reject_non_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "file.png"
    path.write_bytes(b"fake")
    loader = JPEGLoader()
    with pytest.raises(ImageLoadError):
        loader.load(path)
