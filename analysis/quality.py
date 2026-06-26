"""Image quality metric analysis."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from core.models import ImageStatistics, QualityAnalysis
from utils.labeling import quality_label

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """Estimates sharpness, noise, blur, and exposure issues."""

    def analyze(self, rgb: np.ndarray, stats: ImageStatistics) -> QualityAnalysis:
        """Compute quality metrics without altering the image."""
        gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = gray_u8.astype(np.float32)

        laplacian_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
        noise_estimate = float(np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0)))
        blur_score = 1.0 / (laplacian_var + 1e-6)

        over_exp = stats.highlight_percentage
        under_exp = stats.shadow_percentage

        clipped_high = float(np.count_nonzero(gray >= 252) / gray.size * 100)
        clipped_low = float(np.count_nonzero(gray <= 3) / gray.size * 100)
        clipping = clipped_high + clipped_low

        result = QualityAnalysis(
            sharpness=round(laplacian_var, 2),
            noise=round(noise_estimate, 2),
            blur=round(blur_score, 6),
            over_exposure=round(over_exp, 2),
            under_exposure=round(under_exp, 2),
            clipping=round(clipping, 2),
        )

        result.sharpness_label = quality_label(laplacian_var, 50, 200)
        result.noise_label = quality_label(noise_estimate, 3, 8, invert=True)
        result.blur_label = quality_label(blur_score, 0.01, 0.005, invert=True)

        return result
