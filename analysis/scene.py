"""Heuristic scene classification analysis."""

from __future__ import annotations

import logging

import numpy as np

from core.models import (
    BackgroundAnalysis,
    ExifData,
    FaceAnalysis,
    ImageStatistics,
    QualityAnalysis,
    SceneAnalysis,
)

logger = logging.getLogger(__name__)


class SceneAnalyzer:
    """Estimates scene type using heuristics on statistics and content."""

    def analyze(
        self,
        rgb: np.ndarray,
        stats: ImageStatistics,
        exif: ExifData,
        faces: list[FaceAnalysis],
        background: BackgroundAnalysis,
        quality: QualityAnalysis,
    ) -> SceneAnalysis:
        """Classify scene attributes with confidence scoring."""
        scores: dict[str, float] = {}

        brightness = stats.brightness
        green_pct = background.green_percentage
        blue_pct = background.blue_percentage
        face_count = len(faces)
        flash_exif = "flash" in exif.flash.lower() if exif.flash else False

        scores["outdoor"] = self._score_outdoor(green_pct, blue_pct, brightness)
        scores["indoor"] = self._score_indoor(green_pct, brightness, face_count)
        scores["nature"] = self._score_nature(green_pct, blue_pct)
        scores["portrait"] = self._score_portrait(face_count, rgb.shape)
        scores["night"] = self._score_night(brightness, stats.shadow_percentage)
        scores["golden_hour"] = self._score_golden_hour(stats)
        scores["reception"] = self._score_reception(face_count, brightness, background)
        scores["stage"] = self._score_stage(background, quality, brightness)
        scores["backlight"] = self._score_backlight(stats, faces)
        scores["flash"] = self._score_flash(flash_exif, quality, stats)

        scene = SceneAnalysis()
        scene.outdoor = scores["outdoor"] >= 0.5
        scene.indoor = scores["indoor"] >= 0.5
        scene.reception = scores["reception"] >= 0.5
        scene.stage = scores["stage"] >= 0.5
        scene.portrait = scores["portrait"] >= 0.5
        scene.nature = scores["nature"] >= 0.5
        scene.night = scores["night"] >= 0.5
        scene.golden_hour = scores["golden_hour"] >= 0.5
        scene.backlight = scores["backlight"] >= 0.5
        scene.flash = scores["flash"] >= 0.5

        primary, confidence = self._primary_scene(scores)
        scene.primary_scene = primary
        scene.confidence_score = round(confidence * 100, 1)

        return scene

    def _score_outdoor(self, green: float, blue: float, brightness: float) -> float:
        score = 0.0
        if green > 15:
            score += 0.35
        if blue > 10:
            score += 0.25
        if brightness > 100:
            score += 0.25
        if green + blue > 25:
            score += 0.15
        return min(1.0, score)

    def _score_indoor(self, green: float, brightness: float, faces: int) -> float:
        score = 0.0
        if green < 10:
            score += 0.3
        if 80 <= brightness <= 170:
            score += 0.25
        if faces > 0:
            score += 0.2
        if brightness < 130:
            score += 0.15
        return min(1.0, score)

    def _score_nature(self, green: float, blue: float) -> float:
        score = 0.0
        if green > 25:
            score += 0.5
        if blue > 8:
            score += 0.2
        if green > blue:
            score += 0.2
        return min(1.0, score)

    def _score_portrait(self, face_count: int, shape: tuple) -> float:
        if face_count == 0:
            return 0.0
        height, width = shape[0], shape[1]
        face_ratio = face_count / max(1, (width * height) / 500_000)
        return min(1.0, 0.4 + face_count * 0.2 + face_ratio * 0.1)

    def _score_night(self, brightness: float, shadow: float) -> float:
        score = 0.0
        if brightness < 70:
            score += 0.5
        if shadow > 35:
            score += 0.3
        if brightness < 50:
            score += 0.2
        return min(1.0, score)

    def _score_golden_hour(self, stats: ImageStatistics) -> float:
        avg = stats.average_rgb
        r, g, b = avg.get("r", 0), avg.get("g", 0), avg.get("b", 0)
        warmth = r - b
        score = 0.0
        if warmth > 15:
            score += 0.4
        if 90 <= stats.brightness <= 160:
            score += 0.3
        if stats.highlight_percentage < 15:
            score += 0.2
        return min(1.0, score)

    def _score_reception(
        self, faces: int, brightness: float, background: BackgroundAnalysis
    ) -> float:
        score = 0.0
        if faces >= 2:
            score += 0.35
        if background.wood_brown_percentage > 5 or background.white_percentage > 20:
            score += 0.25
        if 90 <= brightness <= 180:
            score += 0.2
        return min(1.0, score)

    def _score_stage(
        self, background: BackgroundAnalysis, quality: QualityAnalysis, brightness: float
    ) -> float:
        score = 0.0
        if background.black_percentage > 30:
            score += 0.35
        if brightness < 100 or quality.over_exposure > 8:
            score += 0.2
        if background.dominant_color in ("Black", "Red", "Magenta"):
            score += 0.25
        return min(1.0, score)

    def _score_backlight(self, stats: ImageStatistics, faces: list[FaceAnalysis]) -> float:
        score = 0.0
        if stats.highlight_percentage > 12:
            score += 0.35
        if faces:
            face_shadows = [f.shadow_percentage for f in faces]
            if face_shadows and float(np.mean(face_shadows)) > 20:
                score += 0.35
        if stats.brightness > 140 and stats.shadow_percentage > 15:
            score += 0.2
        return min(1.0, score)

    def _score_flash(
        self, flash_exif: bool, quality: QualityAnalysis, stats: ImageStatistics
    ) -> float:
        score = 0.0
        if flash_exif:
            score += 0.5
        if stats.highlight_percentage > 8 and stats.brightness > 130:
            score += 0.25
        if quality.over_exposure > 5:
            score += 0.15
        return min(1.0, score)

    def _primary_scene(self, scores: dict[str, float]) -> tuple[str, float]:
        """Return highest scoring scene and its confidence."""
        if not scores:
            return "Unknown", 0.0
        primary = max(scores, key=scores.get)
        confidence = scores[primary]
        return primary.replace("_", " ").title(), confidence
