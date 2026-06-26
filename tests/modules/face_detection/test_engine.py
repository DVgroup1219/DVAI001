"""Tests for detection engine."""

import numpy as np

from src.modules.face_detection.engine import FacePersonDetectionEngine


def test_detect_array_with_mocks(mock_engine, sample_rgb) -> None:
    report = mock_engine.detect_array(sample_rgb, image_name="test.jpg")
    assert report.people == 1
    assert report.face_count == 1
    assert len(report.faces) == 1
    assert report.faces[0].confidence == 0.99
    assert report.faces[0].orientation in ("front", "left", "right", "unknown", "profile")
    assert report.faces[0].bbox.as_list() == [120, 60, 80, 90]


def test_detect_file(sample_jpeg, mock_engine) -> None:
    engine = mock_engine
    report = engine.detect_file(sample_jpeg)
    assert report.image == "IMG001.JPG"
    assert report.people == 1


def test_engine_context_manager(sample_rgb) -> None:
    with FacePersonDetectionEngine() as engine:
        engine._person_detector = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        engine._face_detector = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        engine._person_detector.detect.return_value = []
        engine._face_detector.detect.return_value = []
        report = engine.detect_array(sample_rgb)
        assert report.people == 0
