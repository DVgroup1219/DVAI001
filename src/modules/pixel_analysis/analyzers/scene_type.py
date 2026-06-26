"""Scene type classification from pixel features."""

from __future__ import annotations

import logging

import numpy as np

from src.modules.pixel_analysis.analyzers.base import BasePixelAnalyzer
from src.modules.pixel_analysis.models import (
    ColorCastResult,
    DominantColorResult,
    SceneTypeResult,
    SkinToneRegionResult,
    SkyResult,
    VegetationResult,
)

logger = logging.getLogger(__name__)

SCENE_TYPES = (
    "Outdoor Portrait",
    "Wedding Hall",
    "Indoor Flash",
    "Reception Stage",
    "Studio",
)


class SceneTypeAnalyzer:
    """Classifies scene type using combined pixel analysis features."""

    def classify(
        self,
        rgb: np.ndarray,
        color_cast: ColorCastResult,
        skin: SkinToneRegionResult,
        vegetation: VegetationResult,
        sky: SkyResult,
        dominant: DominantColorResult,
    ) -> SceneTypeResult:
        scores: dict[str, float] = {}
        brightness = float(np.mean(cv2_gray(rgb)))
        contrast = float(np.std(cv2_gray(rgb)))
        skin_pct = skin.skin_pixel_percentage
        veg_pct = vegetation.vegetation_percentage
        sky_pct = sky.sky_percentage
        primary = dominant.primary_dominant_color

        scores["Outdoor Portrait"] = self._score_outdoor_portrait(skin_pct, veg_pct, sky_pct, brightness)
        scores["Wedding Hall"] = self._score_wedding_hall(skin_pct, primary, brightness)
        scores["Indoor Flash"] = self._score_indoor_flash(brightness, color_cast, skin_pct)
        scores["Reception Stage"] = self._score_reception_stage(skin_pct, primary, contrast, brightness)
        scores["Studio"] = self._score_studio(contrast, primary, skin_pct, brightness)

        best = max(scores, key=scores.get)
        confidence = scores[best] * 100
        logger.debug("Scene type: %s (%.0f%%)", best, confidence)
        return SceneTypeResult(scene_type=best, confidence=confidence, scores=scores)

    def _score_outdoor_portrait(self, skin: float, veg: float, sky: float, brightness: float) -> float:
        score = 0.0
        if skin > 3:
            score += 0.35
        if veg > 10 or sky > 8:
            score += 0.35
        if brightness > 90:
            score += 0.2
        if skin > 5 and (veg > 5 or sky > 5):
            score += 0.1
        return min(1.0, score)

    def _score_wedding_hall(self, skin: float, primary: str, brightness: float) -> float:
        score = 0.0
        if skin > 8:
            score += 0.3
        if primary in ("White", "Brown", "Yellow", "Neutral"):
            score += 0.3
        if 80 <= brightness <= 180:
            score += 0.25
        return min(1.0, score)

    def _score_indoor_flash(self, brightness: float, cast: ColorCastResult, skin: float) -> float:
        score = 0.0
        if brightness > 130:
            score += 0.3
        if cast.yellow_cast in ("Low", "Medium", "High"):
            score += 0.2
        if skin > 5:
            score += 0.25
        if cast.magenta_cast == "None":
            score += 0.1
        return min(1.0, score)

    def _score_reception_stage(self, skin: float, primary: str, contrast: float, brightness: float) -> float:
        score = 0.0
        if skin > 10:
            score += 0.3
        if primary in ("Black", "Red", "Magenta", "Blue"):
            score += 0.25
        if contrast > 45:
            score += 0.2
        if brightness < 120:
            score += 0.15
        return min(1.0, score)

    def _score_studio(self, contrast: float, primary: str, skin: float, brightness: float) -> float:
        score = 0.0
        if primary in ("Gray", "White", "Neutral", "Black"):
            score += 0.35
        if 90 <= brightness <= 160:
            score += 0.25
        if contrast < 55:
            score += 0.2
        if skin > 3:
            score += 0.15
        return min(1.0, score)


def cv2_gray(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
