"""Tests for image loader."""

from pathlib import Path

import pytest

from core.image_loader import ImageLoader, ImageLoadError


def test_load_jpeg(sample_jpeg: Path) -> None:
  loader = ImageLoader()
  loaded = loader.load(sample_jpeg)
  assert loaded.rgb_array.shape == (200, 300, 3)
  assert loaded.file_size_bytes > 0


def test_reject_non_jpeg(tmp_path: Path) -> None:
  path = tmp_path / "file.png"
  path.write_bytes(b"not a jpeg")
  loader = ImageLoader()
  with pytest.raises(ImageLoadError):
    loader.load(path)
