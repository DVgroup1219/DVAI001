"""Face detection and skin tone analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from core.models import FaceAnalysis
from utils.color_conversion import lab_to_display_values, rgb_to_hsv, rgb_to_lab

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """Detects faces and analyzes skin regions without modifying the image."""

    HIGHLIGHT_THRESHOLD = 220
    SHADOW_THRESHOLD = 40

    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(str(cascade_path))
        if self._detector.empty():
            logger.warning("Face cascade failed to load")

    def analyze(self, rgb: np.ndarray) -> list[FaceAnalysis]:
        """Detect all faces and compute per-face skin metrics."""
        if self._detector.empty():
            return []

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape

        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        results: list[FaceAnalysis] = []
        for x, y, w, h in faces:
            x2 = min(x + w, width)
            y2 = min(y + h, height)
            face_rgb = rgb[y:y2, x:x2]
            if face_rgb.size == 0:
                continue

            skin_mask = self._skin_mask(face_rgb)
            skin_pixels = face_rgb[skin_mask]
            face_size = int(w * h)
            skin_area = int(np.count_nonzero(skin_mask))

            analysis = FaceAnalysis(
                bounding_box={"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                face_size=face_size,
                skin_area=skin_area,
            )

            if skin_area > 0:
                analysis.average_skin_rgb = self._mean_rgb(skin_pixels)
                lab = rgb_to_lab(skin_pixels.reshape(-1, 1, 3))
                hsv = rgb_to_hsv(skin_pixels.reshape(-1, 1, 3))
                analysis.average_skin_lab = self._mean_lab(lab)
                analysis.average_skin_hsv = self._mean_hsv(hsv)

                l_ch = lab[:, :, 0].astype(np.float32)
                analysis.highlight_percentage = float(
                    np.count_nonzero(l_ch >= self.HIGHLIGHT_THRESHOLD) / skin_area * 100
                )
                analysis.shadow_percentage = float(
                    np.count_nonzero(l_ch <= self.SHADOW_THRESHOLD) / skin_area * 100
                )

            results.append(analysis)

        return results

    def _skin_mask(self, face_rgb: np.ndarray) -> np.ndarray:
        """Build a skin pixel mask using HSV and YCrCb rules."""
        bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

        lower_hsv = np.array([0, 30, 50], dtype=np.uint8)
        upper_hsv = np.array([25, 180, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        return combined.astype(bool)

    def _mean_rgb(self, pixels: np.ndarray) -> dict[str, float]:
        means = np.mean(pixels.reshape(-1, 3), axis=0)
        return {"r": float(means[0]), "g": float(means[1]), "b": float(means[2])}

    def _mean_lab(self, lab: np.ndarray) -> dict[str, float]:
        l_ch, a_ch, b_ch = lab_to_display_values(lab)
        return {
            "l": float(np.mean(l_ch)),
            "a": float(np.mean(a_ch)),
            "b": float(np.mean(b_ch)),
        }

    def _mean_hsv(self, hsv: np.ndarray) -> dict[str, float]:
        flat = hsv.reshape(-1, 3)
        return {
            "h": float(np.mean(flat[:, 0])),
            "s": float(np.mean(flat[:, 1])),
            "v": float(np.mean(flat[:, 2])),
        }
