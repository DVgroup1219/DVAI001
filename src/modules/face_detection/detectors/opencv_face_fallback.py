"""OpenCV Haar cascade fallback for face detection."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.modules.face_detection.detectors.base import BaseDetector, RawFaceDetection
from src.modules.face_detection.models import BoundingBox

logger = logging.getLogger(__name__)


class OpenCVFaceDetector(BaseDetector[RawFaceDetection]):
    """Fallback face detector using OpenCV Haar cascades."""

    backend_name = "opencv_haar"

    def __init__(self, min_neighbors: int = 5, min_size: int = 30) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        if self._cascade.empty():
            raise RuntimeError(f"OpenCV face cascade failed to load: {cascade_path}")

    def detect(self, image_rgb: np.ndarray) -> list[RawFaceDetection]:
        """Detect faces using Haar cascade."""
        if image_rgb.size == 0:
            logger.warning("OpenCV Haar received empty image array")
            return []

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=self._min_neighbors,
            minSize=(self._min_size, self._min_size),
        )

        results: list[RawFaceDetection] = []
        for x, y, w, h in faces:
            results.append(
                RawFaceDetection(
                    bbox=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                    confidence=0.75,
                    keypoints=None,
                )
            )

        logger.debug("OpenCV Haar detected %d faces", len(results))
        return results

    def close(self) -> None:
        """No persistent resources to release."""
