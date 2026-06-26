"""Overall color cast and saturation analysis."""

from __future__ import annotations

import logging

import numpy as np

from core.models import ColorAnalysis, FaceAnalysis, ImageStatistics
from utils.color_conversion import rgb_to_hsv, rgb_to_lab, lab_to_display_values
from utils.labeling import cast_level

logger = logging.getLogger(__name__)


class ColorAnalyzer:
    """Estimates color temperature, casts, and saturation."""

    def analyze(
        self,
        rgb: np.ndarray,
        stats: ImageStatistics,
        faces: list[FaceAnalysis],
    ) -> ColorAnalysis:
        """Compute color analysis metrics."""
        result = ColorAnalysis()

        lab = rgb_to_lab(rgb)
        _, a_ch, b_ch = lab_to_display_values(lab)
        hsv = rgb_to_hsv(rgb)
        saturation = hsv[:, :, 1].astype(np.float32)

        mean_a = float(np.mean(a_ch))
        mean_b = float(np.mean(b_ch))

        warm_score = max(0.0, mean_b)
        cool_score = max(0.0, -mean_b) + max(0.0, -mean_a) * 0.3
        neutral_score = 100.0 - min(100.0, abs(mean_a) + abs(mean_b))

        total = warm_score + cool_score + neutral_score + 1e-6
        result.warm = round(warm_score / total * 100, 2)
        result.cool = round(cool_score / total * 100, 2)
        result.neutral = round(neutral_score / total * 100, 2)

        result.green_cast = cast_level(max(0.0, -mean_a))
        result.magenta_cast = cast_level(max(0.0, mean_a))
        result.yellow_cast = cast_level(max(0.0, mean_b))
        result.blue_cast = cast_level(max(0.0, -mean_b))

        result.overall_saturation = round(float(np.mean(saturation)), 2)
        vibrant_mask = saturation > 80
        result.overall_vibrance = round(
            float(np.count_nonzero(vibrant_mask) / saturation.size * 100), 2
        )

        result.skin_tone_label = self._skin_tone_label(stats, faces)
        return result

    def _skin_tone_label(
        self, stats: ImageStatistics, faces: list[FaceAnalysis]
    ) -> str:
        """Derive skin tone label from face skin or global averages."""
        if faces:
            skin_labs = [f.average_skin_lab for f in faces if f.average_skin_lab]
            if skin_labs:
                avg_a = float(np.mean([s["a"] for s in skin_labs]))
                avg_b = float(np.mean([s["b"] for s in skin_labs]))
            else:
                avg_a = stats.average_lab.get("a", 0.0)
                avg_b = stats.average_lab.get("b", 0.0)
        else:
            avg_a = stats.average_lab.get("a", 0.0)
            avg_b = stats.average_lab.get("b", 0.0)

        warmth = "Warm" if avg_b > 5 else "Cool" if avg_b < -5 else "Neutral"
        chroma = "Vivid" if abs(avg_a) > 8 else "Muted" if abs(avg_a) < 3 else "Neutral"
        if warmth == "Neutral" and chroma == "Neutral":
            return "Neutral"
        if chroma == "Neutral":
            return warmth
        return f"{warmth} {chroma}"
