"""Tests for JSON and CSV exporters."""

import json
from pathlib import Path

from src.modules.pixel_analysis.exporter import export_analysis_csv, export_analysis_json, export_summary_json
from src.modules.pixel_analysis.models import PixelAnalysisReport


def test_export_json_and_csv(tmp_path: Path) -> None:
    reports = [PixelAnalysisReport(image="IMG001.JPG"), PixelAnalysisReport(image="IMG002.JPG")]
    json_path = tmp_path / "IMG001_pixel_analysis.json"
    csv_path = tmp_path / "pixel_analysis.csv"
    summary_path = tmp_path / "summary.json"

    export_analysis_json(reports[0], json_path)
    export_analysis_csv(reports, csv_path)
    export_summary_json(reports, summary_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["image"] == "IMG001.JPG"
    assert "color_cast" in data
    assert "histogram" in data

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "IMG001.JPG" in csv_text
    assert "scene_type" in csv_text
