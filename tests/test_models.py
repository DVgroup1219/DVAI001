"""Tests for data models."""

from core.models import ImageAnalysis


def test_summary_row_contains_key_fields() -> None:
  analysis = ImageAnalysis(file_path="/tmp/test.jpg", file_name="test.jpg")
  row = analysis.summary_row()
  assert row["file_name"] == "test.jpg"
  assert "outdoor" in row
  assert "faces_count" in row
  assert "dominant_color" in row
