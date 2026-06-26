"""Batch processing for Module 03."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from src.modules.pixel_analysis.config import PixelAnalysisConfig
from src.modules.pixel_analysis.engine import PixelAnalysisEngine
from src.modules.pixel_analysis.exporter import (
    export_analysis_csv,
    export_analysis_json,
    export_summary_json,
)
from src.modules.pixel_analysis.loader import JPEGLoader
from src.modules.pixel_analysis.models import PixelAnalysisReport

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class BatchProcessor:
    """Processes single images, folders, or batches of JPEG paths."""

    def __init__(self, config: PixelAnalysisConfig | None = None) -> None:
        self._config = config or PixelAnalysisConfig()
        self._loader = JPEGLoader()
        self._engine = PixelAnalysisEngine(self._config)

    def collect_jpegs_from_folder(self, folder: Path) -> list[Path]:
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        paths = [p.resolve() for p in sorted(folder.rglob("*")) if self._loader.is_jpeg(p)]
        logger.info("Collected %d JPEG files from %s", len(paths), folder)
        return paths

    def process_paths(
        self,
        paths: list[Path],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> list[PixelAnalysisReport]:
        output_dir = self._config.resolved_output_dir()
        json_dir = self._config.json_dir()
        json_dir.mkdir(parents=True, exist_ok=True)

        reports: list[PixelAnalysisReport] = []
        total = len(paths)

        for index, path in enumerate(paths, start=1):
            if progress_callback:
                progress_callback(index, total, path.name)

            report = self._engine.analyze_file(path)
            reports.append(report)

            if report.errors:
                logger.warning("Errors for %s: %s", path.name, "; ".join(report.errors))

            json_path = json_dir / f"{path.stem}_pixel_analysis.json"
            try:
                export_analysis_json(report, json_path)
            except OSError as exc:
                logger.error("JSON export failed for %s: %s", path.name, exc, exc_info=True)
                report.errors.append(f"export_failed: {exc}")

        csv_path = output_dir / self._config.csv_filename
        export_analysis_csv(reports, csv_path)
        export_summary_json(reports, output_dir / self._config.summary_json_filename)
        return reports

    def process_single(self, image_path: Path) -> PixelAnalysisReport:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.process_paths([image_path])[0]

    def process_folder(self, folder: Path) -> list[PixelAnalysisReport]:
        paths = self.collect_jpegs_from_folder(folder)
        if not paths:
            logger.warning("No JPEG files found in %s", folder)
            return []
        return self.process_paths(paths)
