"""Overall color histogram analyzer."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.modules.pixel_analysis.analyzers.base import BasePixelAnalyzer
from src.modules.pixel_analysis.models import HistogramResult

logger = logging.getLogger(__name__)


class HistogramAnalyzer(BasePixelAnalyzer[HistogramResult]):
    """Computes overall RGB and luminance histograms."""

    def analyze(self, rgb: np.ndarray) -> HistogramResult:
        result = HistogramResult()
        if rgb.size == 0:
            return result

        for i, channel in enumerate(("rgb_red", "rgb_green", "rgb_blue")):
            counts, _ = np.histogram(rgb[:, :, i], bins=256, range=(0, 256))
            setattr(result, channel, counts.tolist())

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lum_counts, _ = np.histogram(gray, bins=256, range=(0, 256))
        result.luminance = lum_counts.tolist()
        logger.debug("Histogram computed for %dx%d image", rgb.shape[1], rgb.shape[0])
        return result
