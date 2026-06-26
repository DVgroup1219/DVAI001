"""Tests for labeling helpers."""

from utils.labeling import cast_level, dynamic_range_label, quality_label


def test_cast_level_thresholds() -> None:
  assert cast_level(1.0) == "None"
  assert cast_level(5.0) == "Low"
  assert cast_level(10.0) == "Medium"
  assert cast_level(20.0) == "High"


def test_quality_label() -> None:
  assert quality_label(10, 50, 200) == "Low"
  assert quality_label(100, 50, 200) == "Medium"
  assert quality_label(300, 50, 200) == "High"


def test_dynamic_range_label() -> None:
  assert dynamic_range_label(30, 5, 5) == "Low"
  assert dynamic_range_label(45, 10, 10) == "Medium"
  assert dynamic_range_label(70, 20, 20) == "High"
