"""Read-only JPEG image loading for detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class ImageLoadError(Exception):
    """Raised when a JPEG cannot be loaded."""


@dataclass(frozen=True)
class LoadedJPEG:
    """Read-only loaded JPEG for detection."""

    path: Path
    rgb: np.ndarray
    width: int
    height: int


class JPEGLoader:
    """Loads Sony JPEG images without modifying the source file."""

    JPEG_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

    def is_jpeg(self, path: Path) -> bool:
        """Return True if path is a JPEG file."""
        return path.suffix in self.JPEG_EXTENSIONS and path.is_file()

    def load(self, path: Path) -> LoadedJPEG:
        """Load JPEG into memory for detection only."""
        path = path.resolve()
        if not self.is_jpeg(path):
            raise ImageLoadError(f"Not a JPEG file: {path}")

        try:
            with Image.open(path) as img:
                img.load()
                oriented = ImageOps.exif_transpose(img)
                rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
        except OSError as exc:
            logger.exception("Failed to load JPEG: %s", path)
            raise ImageLoadError(f"Cannot load JPEG: {path}") from exc

        height, width = rgb.shape[:2]
        return LoadedJPEG(path=path, rgb=rgb, width=width, height=height)

    def load_bgr(self, path: Path) -> tuple[LoadedJPEG, np.ndarray]:
        """Load JPEG and return RGB container plus BGR array for OpenCV."""
        loaded = self.load(path)
        bgr = cv2.cvtColor(loaded.rgb, cv2.COLOR_RGB2BGR)
        return loaded, bgr
