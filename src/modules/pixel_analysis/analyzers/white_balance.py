"""White balance estimation analyzer."""

from __future__ import annotations

import logging

import numpy as np

from src.modules.pixel_analysis.analyzers.base import BasePixelAnalyzer
from src.modules.pixel_analysis.models import WhiteBalanceEstimate
from src.modules.pixel_analysis.utils.color_conversion import lab_channels, rgb_to_lab

logger = logging.getLogger(__name__)


class WhiteBalanceAnalyzer(BasePixelAnalyzer[WhiteBalanceEstimate]):
    """Estimates white balance temperature and tint from gray-world LAB stats."""

    def analyze(self, rgb: np.ndarray) -> WhiteBalanceEstimate:
        result = WhiteBalanceEstimate()
        if rgb.size == 0:
            return result

        means = np.mean(rgb.reshape(-1, 3), axis=0)
        result.average_rgb = {
            "r": float(means[0]),
            "g": float(means[1]),
            "b": float(means[2]),
        }

        lab = rgb_to_lab(rgb)
        _, a_ch, b_ch = lab_channels(lab)
        mean_a = float(np.mean(a_ch))
        mean_b = float(np.mean(b_ch))

        result.estimated_temperature_k = self._estimate_temperature(means)
        result.tint_strength = abs(mean_a)
        result.estimated_tint = self._estimate_tint(mean_a, mean_b)

        logger.debug(
            "White balance — %sK tint=%s",
            result.estimated_temperature_k,
            result.estimated_tint,
        )
        return result

    def _estimate_temperature(self, means: np.ndarray) -> float:
        r, g, b = float(means[0]), float(means[1]), float(means[2])
        if b < 1:
            return 6500.0
        ratio = r / b
        if ratio > 1.0:
            temp = 6500.0 - (ratio - 1.0) * 2000.0
        else:
            temp = 6500.0 + (1.0 - ratio) * 2500.0
        return float(max(2500.0, min(10000.0, temp)))

    def _estimate_tint(self, mean_a: float, mean_b: float) -> str:
        if abs(mean_a) < 2 and abs(mean_b) < 2:
            return "Neutral"
        if abs(mean_a) >= abs(mean_b):
            return "Magenta" if mean_a > 0 else "Green"
        return "Yellow" if mean_b > 0 else "Blue"
