"""Tests for pixel analysis data models."""

from src.modules.pixel_analysis.models import ColorCastResult, PixelAnalysisReport


def test_summary_row_has_required_fields() -> None:
    report = PixelAnalysisReport(
        image="IMG001.JPG",
        color_cast=ColorCastResult(green_cast="Medium", blue_cast="Low"),
    )
    row = report.summary_row()
    assert row["file_name"] == "IMG001.JPG"
    assert row["green_cast"] == "Medium"
    assert "scene_type" in row
    assert "vegetation_percentage" in row


def test_to_dict_includes_module_name() -> None:
    report = PixelAnalysisReport(image="test.jpg")
    data = report.to_dict()
    assert "Module 03" in data["module"]
