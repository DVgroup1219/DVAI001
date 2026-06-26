"""Dominant color cluster analyzer."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.modules.pixel_analysis.analyzers.base import BasePixelAnalyzer
from src.modules.pixel_analysis.models import DominantColorCluster, DominantColorResult
from src.modules.pixel_analysis.utils.sampling import sample_pixels

logger = logging.getLogger(__name__)


class DominantColorAnalyzer(BasePixelAnalyzer[DominantColorResult]):
    """Finds dominant color clusters using k-means."""

    def __init__(self, cluster_count: int = 6, sample_size: int = 120_000) -> None:
        self._cluster_count = cluster_count
        self._sample_size = sample_size

    def analyze(self, rgb: np.ndarray) -> DominantColorResult:
        result = DominantColorResult()
        if rgb.size == 0:
            return result

        pixels = rgb.reshape(-1, 3)
        pixels = sample_pixels(pixels, self._sample_size)
        k = min(self._cluster_count, max(1, len(pixels)))

        samples = pixels.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(labels.flatten(), minlength=k)
        order = np.argsort(-counts)

        for idx in order:
            center = centers[idx]
            percentage = float(counts[idx] / len(pixels) * 100)
            name = self._name_color(center)
            result.clusters.append(
                DominantColorCluster(
                    name=name,
                    rgb={
                        "r": round(float(center[0]), 1),
                        "g": round(float(center[1]), 1),
                        "b": round(float(center[2]), 1),
                    },
                    percentage=percentage,
                )
            )

        if result.clusters:
            result.primary_dominant_color = result.clusters[0].name

        logger.debug("Dominant color: %s", result.primary_dominant_color)
        return result

    def _name_color(self, rgb: np.ndarray) -> str:
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if s < 25:
            if v > 200:
                return "White"
            if v < 50:
                return "Black"
            return "Gray"
        if 35 <= h <= 85:
            return "Green"
        if 90 <= h <= 130:
            return "Blue"
        if 8 <= h <= 25:
            return "Brown"
        if h <= 10 or h >= 165:
            return "Red"
        if 15 <= h <= 35:
            return "Yellow"
        if 130 <= h <= 165:
            return "Magenta"
        return "Neutral"
