"""OpenCV HOG fallback for person detection."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.modules.face_detection.detectors.base import BaseDetector, RawPersonDetection
from src.modules.face_detection.models import BoundingBox

logger = logging.getLogger(__name__)


class OpenCVPersonDetector(BaseDetector[RawPersonDetection]):
    """Fallback person detector using OpenCV HOG descriptor."""

    backend_name = "opencv_hog"

    def __init__(self, hit_threshold: float = 0.0) -> None:
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._hit_threshold = hit_threshold

    def detect(self, image_rgb: np.ndarray) -> list[RawPersonDetection]:
        """Detect people using HOG+SVM pedestrian detector."""
        if image_rgb.size == 0:
            logger.warning("OpenCV HOG received empty image array")
            return []

        height, width = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        try:
            rects, weights = self._hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(16, 16),
                scale=1.05,
                hitThreshold=self._hit_threshold,
            )
        except cv2.error as exc:
            logger.exception("OpenCV HOG person detection failed")
            raise RuntimeError("OpenCV person detection failed") from exc

        results: list[RawPersonDetection] = []
        for idx, (x, y, w, h) in enumerate(rects):
            x = max(0, int(x))
            y = max(0, int(y))
            w = min(int(w), width - x)
            h = min(int(h), height - y)
            if w <= 0 or h <= 0:
                continue
            confidence = float(weights[idx][0]) if len(weights) > idx else 0.5
            results.append(
                RawPersonDetection(
                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                    confidence=max(0.0, min(1.0, confidence)),
                )
            )

        logger.debug("OpenCV HOG detected %d people", len(results))
        return results

    def close(self) -> None:
        """No persistent resources to release."""
