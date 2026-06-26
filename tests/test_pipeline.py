"""Integration tests for the analysis pipeline."""

from pathlib import Path

from core.pipeline import AnalysisPipeline


def test_pipeline_analyzes_jpeg(sample_jpeg: Path, output_dir: Path) -> None:
  pipeline = AnalysisPipeline()
  report = pipeline.analyze_paths([sample_jpeg], output_dir)

  assert len(report.images) == 1
  img = report.images[0]
  assert img.file_name == "IMG001.JPG"
  assert img.dimensions.width == 300
  assert img.dimensions.height == 200
  assert img.dimensions.orientation == "Landscape"
  assert img.statistics.brightness > 0

  assert (output_dir / "analysis.json").exists()
  assert (output_dir / "analysis.csv").exists()


def test_collect_paths_from_single_and_folder(
  sample_jpeg: Path, tmp_path: Path
) -> None:
  pipeline = AnalysisPipeline()
  paths = pipeline.collect_paths(single=sample_jpeg, folder=tmp_path)
  assert len(paths) == 1
