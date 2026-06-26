"""Sky percentage analyzer."""

from __future__ import annotations

import logging

import numpy as np

from src.modules.pixel_analysis.analyzers.base import BasePixelAnalyzer
from src.modules.pixel_analysis.models import SkyResult
from src.modules.pixel_analysis.utils.color_conversion import rgb_to_hsv

logger = logging.getLogger(__name__)


class SkyAnalyzer(BasePixelAnalyzer[SkyResult]):
    """Estimates sky coverage from blue hue pixels, weighted toward upper image."""

    def analyze(self, rgb: np.ndarray) -> SkyResult:
        result = SkyResult()
        if rgb.size == 0:
            return result

        height, width = rgb.shape[:2]
        hsv = rgb_to_hsv(rgb)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        sky_mask = (h >= 90) & (h <= 130) & (s >= 25) & (v >= 80)
        upper_half = sky_mask[: height // 2, :]
        result.sky_detected_in_upper_region = bool(np.any(upper_half))

        total = sky_mask.size
        sky_count = int(np.count_nonzero(sky_mask))
        result.sky_percentage = float(sky_count / max(1, total) * 100)

        if sky_count > 0:
            sky_pixels = rgb[sky_mask]
            means = np.mean(sky_pixels.reshape(-1, 3), axis=0)
            result.dominant_sky_rgb = {
                "r": float(means[0]),
                "g": float(means[1]),
                "b": float(means[2]),
            }

        logger.debug("Sky: %.1f%%, upper=%s", result.sky_percentage, result.sky_detected_in_upper_region)
        return result
