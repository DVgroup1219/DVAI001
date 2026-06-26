"""Vegetation percentage analyzer."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.modules.pixel_analysis.analyzers.base import BasePixelAnalyzer
from src.modules.pixel_analysis.models import VegetationResult
from src.modules.pixel_analysis.utils.color_conversion import rgb_to_hsv

logger = logging.getLogger(__name__)


class VegetationAnalyzer(BasePixelAnalyzer[VegetationResult]):
    """Estimates vegetation coverage from green hue pixels."""

    def analyze(self, rgb: np.ndarray) -> VegetationResult:
        result = VegetationResult()
        if rgb.size == 0:
            return result

        hsv = rgb_to_hsv(rgb)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask = (h >= 35) & (h <= 85) & (s >= 40) & (v >= 40)
        total = mask.size
        veg_count = int(np.count_nonzero(mask))
        result.vegetation_percentage = float(veg_count / max(1, total) * 100)

        if veg_count > 0:
            green_pixels = rgb[mask]
            means = np.mean(green_pixels.reshape(-1, 3), axis=0)
            result.dominant_green_rgb = {
                "r": float(means[0]),
                "g": float(means[1]),
                "b": float(means[2]),
            }

        logger.debug("Vegetation: %.1f%%", result.vegetation_percentage)
        return result
