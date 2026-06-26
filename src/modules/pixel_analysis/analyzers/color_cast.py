"""Color cast pixel analyzer."""

from __future__ import annotations

import logging

import numpy as np

from src.modules.pixel_analysis.analyzers.base import BasePixelAnalyzer
from src.modules.pixel_analysis.models import ColorCastResult
from src.modules.pixel_analysis.utils.color_conversion import lab_channels, rgb_to_lab
from src.modules.pixel_analysis.utils.labeling import cast_level

logger = logging.getLogger(__name__)


class ColorCastAnalyzer(BasePixelAnalyzer[ColorCastResult]):
    """Detects green, magenta, blue, and yellow casts from LAB channels."""

    def analyze(self, rgb: np.ndarray) -> ColorCastResult:
        lab = rgb_to_lab(rgb)
        _, a_ch, b_ch = lab_channels(lab)
        mean_a = float(np.mean(a_ch))
        mean_b = float(np.mean(b_ch))

        green_strength = max(0.0, -mean_a)
        magenta_strength = max(0.0, mean_a)
        blue_strength = max(0.0, -mean_b)
        yellow_strength = max(0.0, mean_b)

        result = ColorCastResult(
            green_cast=cast_level(green_strength),
            green_cast_strength=green_strength,
            magenta_cast=cast_level(magenta_strength),
            magenta_cast_strength=magenta_strength,
            blue_cast=cast_level(blue_strength),
            blue_cast_strength=blue_strength,
            yellow_cast=cast_level(yellow_strength),
            yellow_cast_strength=yellow_strength,
        )
        logger.debug(
            "Color cast — green=%s magenta=%s blue=%s",
            result.green_cast,
            result.magenta_cast,
            result.blue_cast,
        )
        return result
