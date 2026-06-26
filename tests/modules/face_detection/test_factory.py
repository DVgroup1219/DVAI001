"""Tests for detector factory fallback logic."""

from unittest.mock import MagicMock, patch

import pytest

from src.modules.face_detection.config import DetectionConfig
from src.modules.face_detection.detectors.factory import create_face_detector, create_person_detector
from src.modules.face_detection.detectors.opencv_face_fallback import OpenCVFaceDetector
from src.modules.face_detection.detectors.opencv_person_fallback import OpenCVPersonDetector
from src.modules.face_detection.exceptions import DetectorUnavailableError


@patch("src.modules.face_detection.detectors.factory.YOLOPersonDetector")
def test_create_person_detector_uses_yolo_when_available(mock_yolo_cls) -> None:
    mock_yolo_cls.return_value = MagicMock()
    detector = create_person_detector(DetectionConfig(), "cpu")
    mock_yolo_cls.assert_called_once()
    assert detector is mock_yolo_cls.return_value


@patch("src.modules.face_detection.detectors.factory.YOLOPersonDetector", side_effect=RuntimeError("no yolo"))
@patch("src.modules.face_detection.detectors.factory.OpenCVPersonDetector")
def test_create_person_detector_falls_back_to_opencv(mock_opencv_cls, _mock_yolo) -> None:
    mock_opencv_cls.return_value = MagicMock(spec=OpenCVPersonDetector)
    detector = create_person_detector(DetectionConfig(), "cpu")
    mock_opencv_cls.assert_called_once()
    assert detector is mock_opencv_cls.return_value


@patch("src.modules.face_detection.detectors.factory.YOLOPersonDetector", side_effect=RuntimeError("no yolo"))
@patch("src.modules.face_detection.detectors.factory.OpenCVPersonDetector", side_effect=RuntimeError("no hog"))
def test_create_person_detector_raises_when_all_backends_fail(_mock_opencv, _mock_yolo) -> None:
    with pytest.raises(DetectorUnavailableError):
        create_person_detector(DetectionConfig(), "cpu")


@patch("src.modules.face_detection.detectors.factory.MediaPipeFaceDetector")
def test_create_face_detector_uses_mediapipe_when_available(mock_mp_cls) -> None:
    mock_mp_cls.return_value = MagicMock()
    detector = create_face_detector(DetectionConfig())
    mock_mp_cls.assert_called_once()
    assert detector is mock_mp_cls.return_value


@patch("src.modules.face_detection.detectors.factory.MediaPipeFaceDetector", side_effect=RuntimeError("no mp"))
@patch("src.modules.face_detection.detectors.factory.OpenCVFaceDetector")
def test_create_face_detector_falls_back_to_opencv(mock_opencv_cls, _mock_mp) -> None:
    mock_opencv_cls.return_value = MagicMock(spec=OpenCVFaceDetector)
    detector = create_face_detector(DetectionConfig())
    mock_opencv_cls.assert_called_once()
    assert detector is mock_opencv_cls.return_value
