"""Tests for batch processor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.modules.face_detection.batch import BatchProcessor
from src.modules.face_detection.config import DetectionConfig
from src.modules.face_detection.models import DetectionReport


def test_collect_jpegs_from_folder(sample_jpeg: Path) -> None:
    processor = BatchProcessor()
    paths = processor.collect_jpegs_from_folder(sample_jpeg.parent)
    assert len(paths) == 1


@patch("src.modules.face_detection.batch.FacePersonDetectionEngine")
def test_process_paths_exports_json(mock_engine_cls, sample_jpeg: Path, tmp_path: Path) -> None:
    mock_engine = MagicMock()
    mock_engine.__enter__.return_value = mock_engine
    mock_engine.detect_file.return_value = DetectionReport(
        image=sample_jpeg.name,
        people=0,
        face_count=0,
    )
    mock_engine_cls.return_value = mock_engine

    config = DetectionConfig(output_dir=tmp_path / "out")
    processor = BatchProcessor(config)
    reports = processor.process_paths([sample_jpeg])

    assert len(reports) == 1
    assert (tmp_path / "out" / "detections" / "IMG001_detection.json").exists()
    assert (tmp_path / "out" / "detections_summary.json").exists()
