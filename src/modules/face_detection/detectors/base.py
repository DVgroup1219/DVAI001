"""Base detector interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from src.modules.face_detection.models import BoundingBox

T = TypeVar("T")


class BaseDetector(ABC, Generic[T]):
    """Abstract detector contract for reusable future modules."""

    backend_name: str = "unknown"

    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> list[T]:
        """Detect objects in an RGB image without modifying it."""

    def close(self) -> None:
        """Release backend resources. Override when needed."""


@dataclass
class RawFaceDetection:
    """Internal face detection with optional MediaPipe keypoints."""

    bbox: BoundingBox
    confidence: float
    keypoints: dict[str, tuple[float, float]] | None = None


@dataclass
class RawPersonDetection:
    """Internal person detection from YOLO or fallback."""

    bbox: BoundingBox
    confidence: float
