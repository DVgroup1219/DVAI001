"""Main pixel analysis engine."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.modules.pixel_analysis.analyzers import (
    ColorCastAnalyzer,
    DominantColorAnalyzer,
    HistogramAnalyzer,
    SceneTypeAnalyzer,
    SkinToneAnalyzer,
    SkyAnalyzer,
    VegetationAnalyzer,
    WhiteBalanceAnalyzer,
)
from src.modules.pixel_analysis.config import PixelAnalysisConfig
from src.modules.pixel_analysis.exceptions import ImageLoadError, ImageValidationError
from src.modules.pixel_analysis.loader import JPEGLoader
from src.modules.pixel_analysis.models import PixelAnalysisReport

logger = logging.getLogger(__name__)


class PixelAnalysisEngine:
    """
    Analyzes actual image pixels without modification or color correction.

    Read-only pixel analysis for Sony JPEG photographs.
    """

    def __init__(self, config: PixelAnalysisConfig | None = None) -> None:
        self._config = config or PixelAnalysisConfig()
        self._loader = JPEGLoader()
        self._color_cast = ColorCastAnalyzer()
        self._skin_tone = SkinToneAnalyzer()
        self._vegetation = VegetationAnalyzer()
        self._sky = SkyAnalyzer()
        self._histogram = HistogramAnalyzer()
        self._white_balance = WhiteBalanceAnalyzer()
        self._dominant_colors = DominantColorAnalyzer(
            cluster_count=self._config.dominant_cluster_count,
            sample_size=self._config.sample_size,
        )
        self._scene_type = SceneTypeAnalyzer()

    @property
    def config(self) -> PixelAnalysisConfig:
        return self._config

    def _validate_rgb(self, rgb: np.ndarray, image_name: str) -> None:
        if not isinstance(rgb, np.ndarray):
            raise ImageValidationError(f"Expected numpy array for {image_name}")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ImageValidationError(
                f"Expected HxWx3 RGB array for {image_name}, got {rgb.shape}"
            )
        if rgb.size == 0:
            raise ImageValidationError(f"Empty image array for {image_name}")

    def analyze_array(self, rgb: np.ndarray, image_name: str = "image.jpg") -> PixelAnalysisReport:
        """Run full pixel analysis on an in-memory RGB image."""
        report = PixelAnalysisReport(image=image_name)
        report.height, report.width = rgb.shape[:2]

        try:
            self._validate_rgb(rgb, image_name)
        except ImageValidationError as exc:
            report.errors.append(str(exc))
            logger.error("Validation failed for %s: %s", image_name, exc)
            return report

        analyzers = [
            ("color_cast", lambda: self._color_cast.analyze(rgb)),
            ("skin_tone", lambda: self._skin_tone.analyze(rgb)),
            ("vegetation", lambda: self._vegetation.analyze(rgb)),
            ("sky", lambda: self._sky.analyze(rgb)),
            ("histogram", lambda: self._histogram.analyze(rgb)),
            ("white_balance", lambda: self._white_balance.analyze(rgb)),
            ("dominant_colors", lambda: self._dominant_colors.analyze(rgb)),
        ]

        for name, fn in analyzers:
            try:
                result = fn()
                setattr(report, name, result)
            except Exception as exc:
                msg = f"{name}_failed: {exc}"
                report.errors.append(msg)
                logger.error("Analyzer '%s' failed for %s: %s", name, image_name, exc, exc_info=True)

        try:
            report.scene_type = self._scene_type.classify(
                rgb,
                report.color_cast,
                report.skin_tone,
                report.vegetation,
                report.sky,
                report.dominant_colors,
            )
        except Exception as exc:
            report.errors.append(f"scene_type_failed: {exc}")
            logger.error("Scene classification failed for %s: %s", image_name, exc, exc_info=True)

        if report.errors:
            logger.warning(
                "Pixel analysis completed with errors for %s: %s",
                image_name,
                "; ".join(report.errors),
            )
        else:
            logger.info(
                "Pixel analysis complete for %s — scene=%s vegetation=%.1f%% sky=%.1f%%",
                image_name,
                report.scene_type.scene_type,
                report.vegetation.vegetation_percentage,
                report.sky.sky_percentage,
            )
        return report

    def analyze_file(self, path: Path) -> PixelAnalysisReport:
        """Load a JPEG and run pixel analysis."""
        try:
            loaded = self._loader.load(path)
        except ImageLoadError as exc:
            report = PixelAnalysisReport(
                image=path.name,
                image_path=str(path.resolve()),
                errors=[f"load_failed: {exc}"],
            )
            logger.error("Cannot load %s: %s", path, exc)
            return report

        report = self.analyze_array(loaded.rgb, image_name=loaded.path.name)
        report.image_path = str(loaded.path)
        report.width = loaded.width
        report.height = loaded.height
        return report
