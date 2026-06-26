"""Tests for exporters."""

import json
from pathlib import Path

from core.models import AnalysisReport, ImageAnalysis
from utils.exporters import export_csv, export_json


def test_export_json_and_csv(tmp_path: Path) -> None:
  report = AnalysisReport(
    images=[ImageAnalysis(file_path="/a/IMG001.JPG", file_name="IMG001.JPG")],
    output_directory=str(tmp_path),
  )

  json_path = tmp_path / "analysis.json"
  csv_path = tmp_path / "analysis.csv"

  export_json(report, json_path)
  export_csv(report, csv_path)

  assert json_path.exists()
  assert csv_path.exists()

  data = json.loads(json_path.read_text(encoding="utf-8"))
  assert data["image_count"] == 1
  assert data["images"][0]["file_name"] == "IMG001.JPG"

  csv_text = csv_path.read_text(encoding="utf-8")
  assert "IMG001.JPG" in csv_text
  assert "file_name" in csv_text
