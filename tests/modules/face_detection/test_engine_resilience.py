"""Tests for engine resilience and error handling."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.modules.face_detection.engine import FacePersonDetectionEngine
from src.modules.face_detection.exceptions import DetectorUnavailableError


def test_detect_array_rejects_invalid_shape() -> None:
    engine = FacePersonDetectionEngine()
    bad = np.zeros((100, 100), dtype=np.uint8)
    report = engine.detect_array(bad, image_name="bad.jpg")
    assert report.errors
    assert report.people == 0
    assert report.face_count == 0


def test_detect_array_continues_when_person_detection_fails(sample_rgb) -> None:
    engine = FacePersonDetectionEngine()
    face_detector = MagicMock()
    face_detector.detect.return_value = []
    person_detector = MagicMock()
    person_detector.detect.side_effect = RuntimeError("yolo inference failed")
    engine._person_detector = person_detector
    engine._face_detector = face_detector

    report = engine.detect_array(sample_rgb, image_name="test.jpg")
    assert any("person_detection_failed" in e for e in report.errors)
    face_detector.detect.assert_called_once()


def test_detect_array_continues_when_face_detection_fails(sample_rgb) -> None:
    engine = FacePersonDetectionEngine()
    face_detector = MagicMock()
    face_detector.detect.side_effect = RuntimeError("mediapipe failed")
    person_detector = MagicMock()
    person_detector.detect.return_value = []
    engine._person_detector = person_detector
    engine._face_detector = face_detector

    report = engine.detect_array(sample_rgb, image_name="test.jpg")
    assert any("face_detection_failed" in e for e in report.errors)
    person_detector.detect.assert_called_once()


@patch("src.modules.face_detection.engine.create_person_detector")
@patch("src.modules.face_detection.engine.create_face_detector")
def test_engine_uses_factory_for_detectors(mock_face_factory, mock_person_factory, sample_rgb) -> None:
    mock_person_factory.return_value = MagicMock(detect=MagicMock(return_value=[]))
    mock_face_factory.return_value = MagicMock(detect=MagicMock(return_value=[]))
    engine = FacePersonDetectionEngine()
    engine.detect_array(sample_rgb)
    mock_person_factory.assert_called_once()
    mock_face_factory.assert_called_once()


def test_engine_logs_unavailable_person_detector(sample_rgb) -> None:
    engine = FacePersonDetectionEngine()
    person_detector = MagicMock()
    person_detector.detect.side_effect = DetectorUnavailableError("no backend")
    engine._person_detector = person_detector
    engine._face_detector = MagicMock(detect=MagicMock(return_value=[]))

    report = engine.detect_array(sample_rgb)
    assert any("person_detection_unavailable" in e for e in report.errors)
