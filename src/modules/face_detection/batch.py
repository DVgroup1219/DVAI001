"""Batch and folder processing for Module 02."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from src.modules.face_detection.config import DetectionConfig
from src.modules.face_detection.engine import FacePersonDetectionEngine
from src.modules.face_detection.exporter import export_batch_summary, export_detection_json
from src.modules.face_detection.loader import JPEGLoader
from src.modules.face_detection.models import DetectionReport
from src.modules.face_detection.preview import PreviewGenerator

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class BatchProcessor:
    """Processes single images, folders, or batches of JPEG paths."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self._config = config or DetectionConfig()
        self._loader = JPEGLoader()
        self._preview = PreviewGenerator()

    def collect_jpegs_from_folder(self, folder: Path) -> list[Path]:
        """Recursively collect JPEG files from a folder."""
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        paths = [
            p.resolve()
            for p in sorted(folder.rglob("*"))
            if self._loader.is_jpeg(p)
        ]
        logger.info("Collected %d JPEG files from %s", len(paths), folder)
        return paths

    def process_paths(
        self,
        paths: list[Path],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> list[DetectionReport]:
        """Process a list of JPEG paths and export JSON results."""
        output_dir = self._config.resolved_output_dir()
        json_dir = self._config.json_dir()
        preview_dir = self._config.preview_dir()
        json_dir.mkdir(parents=True, exist_ok=True)

        reports: list[DetectionReport] = []
        total = len(paths)

        with FacePersonDetectionEngine(self._config) as engine:
            for index, path in enumerate(paths, start=1):
                if progress_callback:
                    progress_callback(index, total, path.name)

                report = engine.detect_file(path)
                reports.append(report)

                json_path = json_dir / f"{path.stem}_detection.json"
                export_detection_json(report, json_path)

                if self._config.preview_enabled and not report.errors:
                    try:
                        loaded = self._loader.load(path)
                        preview_path = preview_dir / f"{path.stem}_preview.jpg"
                        self._preview.generate(loaded.rgb, report, preview_path)
                    except Exception as exc:
                        logger.warning("Preview failed for %s: %s", path.name, exc)

        export_batch_summary(reports, output_dir / "detections_summary.json")
        return reports

    def process_single(self, image_path: Path) -> DetectionReport:
        """Process one JPEG and export JSON."""
        return self.process_paths([image_path])[0]

    def process_folder(self, folder: Path) -> list[DetectionReport]:
        """Process all JPEGs in a folder."""
        paths = self.collect_jpegs_from_folder(folder)
        if not paths:
            logger.warning("No JPEG files found in %s", folder)
            return []
        return self.process_paths(paths)
