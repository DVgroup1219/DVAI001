"""Main analysis pipeline orchestrating all analyzers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from analysis.background import BackgroundAnalyzer
from analysis.color import ColorAnalyzer
from analysis.faces import FaceAnalyzer
from analysis.metadata import MetadataAnalyzer
from analysis.quality import QualityAnalyzer
from analysis.scene import SceneAnalyzer
from analysis.statistics import StatisticsAnalyzer
from core.image_loader import ImageLoader, ImageLoadError
from core.models import AnalysisReport, ImageAnalysis
from utils.exporters import export_csv, export_json
from utils.file_scanner import FileScanner

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class AnalysisPipeline:
    """Orchestrates read-only Sony JPEG analysis."""

    def __init__(self) -> None:
        self._loader = ImageLoader()
        self._scanner = FileScanner()
        self._metadata = MetadataAnalyzer()
        self._statistics = StatisticsAnalyzer()
        self._scene = SceneAnalyzer()
        self._faces = FaceAnalyzer()
        self._background = BackgroundAnalyzer()
        self._color = ColorAnalyzer()
        self._quality = QualityAnalyzer()

    def collect_paths(
        self,
        single: Optional[Path] = None,
        multiple: Optional[list[Path]] = None,
        folder: Optional[Path] = None,
    ) -> list[Path]:
        """Collect JPEG paths from single, multiple, or folder selection."""
        paths: list[Path] = []

        if single is not None:
            paths.append(single)
        if multiple:
            paths.extend(multiple)
        if folder is not None:
            paths.extend(self._scanner.scan_folder(folder))

        return self._scanner.deduplicate(paths)

    def analyze_image(self, path: Path) -> ImageAnalysis:
        """Run full analysis on a single JPEG without modifying it."""
        result = ImageAnalysis(file_path=str(path), file_name=path.name)

        try:
            loaded = self._loader.load(path)
        except ImageLoadError as exc:
            result.errors.append(str(exc))
            logger.error("Skipping %s: %s", path, exc)
            return result

        result.dimensions = self._metadata.analyze_dimensions(loaded)
        result.exif = self._metadata.analyze_exif(loaded)
        result.is_sony_jpeg = self._metadata.is_sony_camera(result.exif)

        stats = self._statistics.analyze(loaded.rgb_array)
        result.statistics = stats

        faces = self._faces.analyze(loaded.rgb_array)
        result.faces = faces

        result.background = self._background.analyze(loaded.rgb_array, faces)
        result.color = self._color.analyze(loaded.rgb_array, stats, faces)
        result.quality = self._quality.analyze(loaded.rgb_array, stats)
        result.scene = self._scene.analyze(
            loaded.rgb_array,
            stats,
            result.exif,
            faces,
            result.background,
            result.quality,
        )

        return result

    def analyze_paths(
        self,
        paths: list[Path],
        output_dir: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> AnalysisReport:
        """Analyze multiple JPEGs and export results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report = AnalysisReport(output_directory=str(output_dir))

        total = len(paths)
        for index, path in enumerate(paths, start=1):
            if progress_callback:
                progress_callback(index, total, path.name)

            logger.info("Analyzing (%d/%d): %s", index, total, path.name)
            report.images.append(self.analyze_image(path))

        export_json(report, output_dir / "analysis.json")
        export_csv(report, output_dir / "analysis.csv")
        logger.info("Exported analysis to %s", output_dir)
        return report
