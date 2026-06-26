"""Integration tests for Module 03 batch processing."""

from pathlib import Path
from unittest.mock import patch

from src.modules.pixel_analysis.batch import BatchProcessor
from src.modules.pixel_analysis.config import PixelAnalysisConfig


def test_process_paths_exports_files(sample_jpeg: Path, tmp_path: Path) -> None:
    config = PixelAnalysisConfig(output_dir=tmp_path / "out")
    processor = BatchProcessor(config)
    reports = processor.process_paths([sample_jpeg])

    assert len(reports) == 1
    out = tmp_path / "out"
    assert (out / "analysis" / "IMG001_pixel_analysis.json").exists()
    assert (out / "pixel_analysis.csv").exists()
    assert (out / "pixel_analysis_summary.json").exists()


def test_collect_jpegs_from_folder(sample_jpeg: Path) -> None:
    processor = BatchProcessor()
    paths = processor.collect_jpegs_from_folder(sample_jpeg.parent)
    assert len(paths) == 1
