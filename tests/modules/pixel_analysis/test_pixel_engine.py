"""Tests for pixel analysis engine."""

import numpy as np

from src.modules.pixel_analysis.engine import PixelAnalysisEngine


def test_analyze_array(sample_rgb) -> None:
    engine = PixelAnalysisEngine()
    report = engine.analyze_array(sample_rgb, "test.jpg")
    assert report.width == 320
    assert report.height == 240
    assert report.scene_type.scene_type != ""
    assert len(report.histogram.rgb_red) == 256
    assert not report.errors


def test_analyze_file(sample_jpeg) -> None:
    engine = PixelAnalysisEngine()
    report = engine.analyze_file(sample_jpeg)
    assert report.image == "IMG001.JPG"
    assert report.color_cast.green_cast in ("None", "Low", "Medium", "High")


def test_invalid_image_rejected() -> None:
    engine = PixelAnalysisEngine()
    bad = np.zeros((100, 100), dtype=np.uint8)
    report = engine.analyze_array(bad, "bad.jpg")
    assert report.errors
