"""Image statistics and histogram analysis."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from core.models import HistogramData, ImageStatistics
from utils.color_conversion import lab_to_display_values, rgb_to_hsv, rgb_to_lab
from utils.labeling import dynamic_range_label

logger = logging.getLogger(__name__)


class StatisticsAnalyzer:
    """Computes pixel statistics and histograms without modifying the image."""

    HIGHLIGHT_THRESHOLD = 240
    SHADOW_THRESHOLD = 15

    def analyze(self, rgb: np.ndarray) -> ImageStatistics:
        """Compute full image statistics from RGB array."""
        stats = ImageStatistics()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lab = rgb_to_lab(rgb)
        hsv = rgb_to_hsv(rgb)

        stats.average_rgb = self._mean_rgb(rgb)
        stats.average_lab = self._mean_lab(lab)
        stats.average_hsv = self._mean_hsv(hsv)
        stats.histogram = self._compute_histograms(rgb, lab, hsv)

        stats.mean = float(np.mean(gray))
        stats.median = float(np.median(gray))
        stats.standard_deviation = float(np.std(gray))
        stats.brightness = stats.mean
        stats.contrast = stats.standard_deviation

        stats.highlight_percentage = float(
            np.count_nonzero(gray >= self.HIGHLIGHT_THRESHOLD) / gray.size * 100
        )
        stats.shadow_percentage = float(
            np.count_nonzero(gray <= self.SHADOW_THRESHOLD) / gray.size * 100
        )
        stats.dynamic_range_estimate = dynamic_range_label(
            stats.standard_deviation,
            stats.highlight_percentage,
            stats.shadow_percentage,
        )

        return stats

    def _mean_rgb(self, rgb: np.ndarray) -> dict[str, float]:
        means = np.mean(rgb.reshape(-1, 3), axis=0)
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

    def _compute_histograms(
        self, rgb: np.ndarray, lab: np.ndarray, hsv: np.ndarray
    ) -> HistogramData:
        hist = HistogramData()

        for i, channel in enumerate(("red", "green", "blue")):
            counts, _ = np.histogram(rgb[:, :, i], bins=256, range=(0, 256))
            setattr(hist, channel, counts.tolist())

        l_ch, a_ch, b_ch = lab_to_display_values(lab)
        hist.l_channel, _ = np.histogram(l_ch, bins=256, range=(0, 255))
        hist.l_channel = hist.l_channel.tolist()

        hist.a_channel, _ = np.histogram(a_ch + 128, bins=256, range=(0, 255))
        hist.a_channel = hist.a_channel.tolist()

        hist.b_channel, _ = np.histogram(b_ch + 128, bins=256, range=(0, 255))
        hist.b_channel = hist.b_channel.tolist()

        hist.h_channel, _ = np.histogram(hsv[:, :, 0], bins=180, range=(0, 180))
        hist.h_channel = hist.h_channel.tolist()

        hist.s_channel, _ = np.histogram(hsv[:, :, 1], bins=256, range=(0, 256))
        hist.s_channel = hist.s_channel.tolist()

        hist.v_channel, _ = np.histogram(hsv[:, :, 2], bins=256, range=(0, 256))
        hist.v_channel = hist.v_channel.tolist()

        return hist
