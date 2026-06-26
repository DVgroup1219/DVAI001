"""High-level pipeline for Module 02."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

from src.modules.face_detection.batch import BatchProcessor
from src.modules.face_detection.config import DetectionConfig
from src.modules.face_detection.models import DetectionReport

logger = logging.getLogger(__name__)


class FaceDetectionPipeline:
    """Facade for single, folder, and batch detection workflows."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self._config = config or DetectionConfig()
        self._processor = BatchProcessor(self._config)

    def run_single(self, image_path: Path, preview: bool = False) -> DetectionReport:
        """Analyze one JPEG."""
        if preview:
            self._config = replace(self._config, preview_enabled=True)
            self._processor = BatchProcessor(self._config)
        return self._processor.process_single(image_path)

    def run_folder(self, folder: Path, preview: bool = False) -> list[DetectionReport]:
        """Analyze all JPEGs in a folder."""
        if preview:
            self._config = replace(self._config, preview_enabled=True)
            self._processor = BatchProcessor(self._config)
        return self._processor.process_folder(folder)

    def run_batch(
        self,
        paths: list[Path],
        preview: bool = False,
    ) -> list[DetectionReport]:
        """Analyze a custom list of JPEG paths."""
        if preview:
            self._config = replace(self._config, preview_enabled=True)
            self._processor = BatchProcessor(self._config)
        return self._processor.process_paths(paths)
