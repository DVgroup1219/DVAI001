"""High-level pipeline for Module 03."""

from __future__ import annotations

from pathlib import Path

from src.modules.pixel_analysis.batch import BatchProcessor
from src.modules.pixel_analysis.config import PixelAnalysisConfig
from src.modules.pixel_analysis.models import PixelAnalysisReport


class PixelAnalysisPipeline:
    """Facade for single, folder, and batch pixel analysis."""

    def __init__(self, config: PixelAnalysisConfig | None = None) -> None:
        self._config = config or PixelAnalysisConfig()
        self._processor = BatchProcessor(self._config)

    def run_single(self, image_path: Path) -> PixelAnalysisReport:
        return self._processor.process_single(image_path)

    def run_folder(self, folder: Path) -> list[PixelAnalysisReport]:
        return self._processor.process_folder(folder)

    def run_batch(self, paths: list[Path]) -> list[PixelAnalysisReport]:
        return self._processor.process_paths(paths)
