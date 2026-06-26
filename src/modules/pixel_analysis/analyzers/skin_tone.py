"""Skin tone region pixel analyzer."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.modules.pixel_analysis.analyzers.base import BasePixelAnalyzer
from src.modules.pixel_analysis.models import SkinToneRegionResult
from src.modules.pixel_analysis.utils.color_conversion import lab_channels, rgb_to_lab

logger = logging.getLogger(__name__)


class SkinToneAnalyzer(BasePixelAnalyzer[SkinToneRegionResult]):
    """Detects skin tone regions using HSV and YCrCb rules."""

    def analyze(self, rgb: np.ndarray) -> SkinToneRegionResult:
        result = SkinToneRegionResult()
        if rgb.size == 0:
            return result

        mask = self._skin_mask(rgb)
        total_pixels = rgb.shape[0] * rgb.shape[1]
        skin_pixels = rgb[mask]
        skin_count = int(np.count_nonzero(mask))

        result.skin_pixel_percentage = float(skin_count / max(1, total_pixels) * 100)
        result.skin_region_count = self._count_regions(mask)

        if skin_count == 0:
            result.skin_tone_label = "No Skin Detected"
            return result

        result.average_skin_rgb = self._mean_rgb(skin_pixels)
        lab = rgb_to_lab(skin_pixels.reshape(-1, 1, 3))
        _, a_ch, b_ch = lab_channels(lab)
        result.average_skin_lab = {
            "l": float(np.mean(lab[:, :, 0])),
            "a": float(np.mean(a_ch)),
            "b": float(np.mean(b_ch)),
        }
        result.skin_tone_label = self._label_skin_tone(
            result.average_skin_lab["a"], result.average_skin_lab["b"]
        )
        result.regions = self._extract_regions(mask)
        logger.debug(
            "Skin tone — %.1f%% pixels, %d regions, label=%s",
            result.skin_pixel_percentage,
            result.skin_region_count,
            result.skin_tone_label,
        )
        return result

    def _skin_mask(self, rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        mask_hsv = cv2.inRange(hsv, np.array([0, 30, 50], np.uint8), np.array([25, 180, 255], np.uint8))
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8))
        return cv2.bitwise_and(mask_hsv, mask_ycrcb).astype(bool)

    def _count_regions(self, mask: np.ndarray) -> int:
        binary = mask.astype(np.uint8) * 255
        num_labels, _ = cv2.connectedComponents(binary)
        return max(0, num_labels - 1)

    def _extract_regions(self, mask: np.ndarray, max_regions: int = 10) -> list[dict]:
        binary = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_regions]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = int(cv2.contourArea(contour))
            if area < 100:
                continue
            regions.append({"bbox": [int(x), int(y), int(w), int(h)], "area": area})
        return regions

    def _mean_rgb(self, pixels: np.ndarray) -> dict[str, float]:
        means = np.mean(pixels.reshape(-1, 3), axis=0)
        return {"r": float(means[0]), "g": float(means[1]), "b": float(means[2])}

    def _label_skin_tone(self, avg_a: float, avg_b: float) -> str:
        warmth = "Warm" if avg_b > 5 else "Cool" if avg_b < -5 else "Neutral"
        undertone = "Pink" if avg_a > 5 else "Olive" if avg_a < -5 else "Neutral"
        if warmth == "Neutral" and undertone == "Neutral":
            return "Neutral"
        return f"{warmth} {undertone}"
