"""Read-only JPEG loader for Module 03."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.modules.pixel_analysis.exceptions import ImageLoadError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedJPEG:
    """Read-only loaded JPEG pixels."""

    path: Path
    rgb: np.ndarray
    width: int
    height: int


class JPEGLoader:
    """Loads JPEG images without modifying source files."""

    JPEG_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

    def is_jpeg(self, path: Path) -> bool:
        return path.suffix in self.JPEG_EXTENSIONS and path.is_file()

    def load(self, path: Path) -> LoadedJPEG:
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
