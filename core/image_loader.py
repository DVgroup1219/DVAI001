"""Read-only JPEG image loading utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedImage:
    """Container for a read-only loaded image and metadata."""

    path: Path
    pil_image: Image.Image
    rgb_array: np.ndarray
    file_size_bytes: int


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded for analysis."""


class ImageLoader:
    """Loads JPEG images in read-only mode without modification."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

    def is_jpeg(self, path: Path) -> bool:
        """Return True if the path has a JPEG extension."""
        return path.suffix in self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> LoadedImage:
        """
        Load a JPEG image without editing pixel data.

        EXIF orientation is applied only for correct analysis dimensions;
        the original file on disk is never modified.
        """
        if not path.exists():
            raise ImageLoadError(f"File not found: {path}")

        if not self.is_jpeg(path):
            raise ImageLoadError(f"Not a JPEG file: {path}")

        try:
            with Image.open(path) as img:
                img.load()
                oriented = ImageOps.exif_transpose(img)
                rgb = oriented.convert("RGB")
                array = np.asarray(rgb, dtype=np.uint8)
                file_size = path.stat().st_size
                return LoadedImage(
                    path=path,
                    pil_image=rgb.copy(),
                    rgb_array=array,
                    file_size_bytes=file_size,
                )
        except OSError as exc:
            logger.exception("Failed to load image: %s", path)
            raise ImageLoadError(f"Cannot load image: {path}") from exc
