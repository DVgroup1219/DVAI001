"""Background color composition analysis."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from core.models import BackgroundAnalysis, FaceAnalysis

logger = logging.getLogger(__name__)


class BackgroundAnalyzer:
    """Estimates background color distribution excluding face regions."""

    def analyze(
        self, rgb: np.ndarray, faces: list[FaceAnalysis], sample_size: int = 120_000
    ) -> BackgroundAnalysis:
        """Estimate dominant and categorical background colors."""
        mask = np.ones(rgb.shape[:2], dtype=bool)
        for face in faces:
            box = face.bounding_box
            x, y = box.get("x", 0), box.get("y", 0)
            w, h = box.get("width", 0), box.get("height", 0)
            mask[y : y + h, x : x + w] = False

        pixels = rgb[mask]
        if pixels.size == 0:
            pixels = rgb.reshape(-1, 3)

        if len(pixels) > sample_size:
            indices = np.random.default_rng(42).choice(len(pixels), sample_size, replace=False)
            pixels = pixels[indices]

        result = BackgroundAnalysis()
        result.dominant_colors = self._dominant_colors(pixels)
        result.dominant_color = result.dominant_colors[0]["name"] if result.dominant_colors else ""

        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        total = len(pixels)

        result.green_percentage = self._color_percentage(hsv, "green", total)
        result.blue_percentage = self._color_percentage(hsv, "blue", total)
        result.white_percentage = self._color_percentage(hsv, "white", total)
        result.black_percentage = self._color_percentage(hsv, "black", total)
        result.wood_brown_percentage = self._color_percentage(hsv, "brown", total)

        return result

    def _dominant_colors(self, pixels: np.ndarray, k: int = 5) -> list[dict]:
        """K-means cluster dominant colors."""
        if len(pixels) < k:
            k = max(1, len(pixels))

        samples = pixels.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(labels.flatten(), minlength=k)
        order = np.argsort(-counts)

        colors: list[dict] = []
        for idx in order:
            center = centers[idx]
            percentage = float(counts[idx] / len(pixels) * 100)
            name = self._name_color(center)
            colors.append(
                {
                    "name": name,
                    "rgb": {
                        "r": round(float(center[0]), 1),
                        "g": round(float(center[1]), 1),
                        "b": round(float(center[2]), 1),
                    },
                    "percentage": round(percentage, 2),
                }
            )
        return colors

    def _color_percentage(self, hsv: np.ndarray, category: str, total: int) -> float:
        """Estimate percentage of pixels matching a color category."""
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

        if category == "green":
            mask = ((h >= 35) & (h <= 85) & (s >= 40) & (v >= 40))
        elif category == "blue":
            mask = ((h >= 90) & (h <= 130) & (s >= 40) & (v >= 40))
        elif category == "white":
            mask = (s <= 30) & (v >= 200)
        elif category == "black":
            mask = v <= 40
        elif category == "brown":
            mask = ((h >= 8) & (h <= 25) & (s >= 30) & (v >= 30) & (v <= 180))
        else:
            mask = np.zeros(len(hsv), dtype=bool)

        return round(float(np.count_nonzero(mask) / total * 100), 2)

    def _name_color(self, rgb: np.ndarray) -> str:
        """Map RGB center to a simple color name."""
        r, g, b = rgb
        hsv = cv2.cvtColor(
            np.uint8([[rgb]]), cv2.COLOR_RGB2HSV
        )[0][0]
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
