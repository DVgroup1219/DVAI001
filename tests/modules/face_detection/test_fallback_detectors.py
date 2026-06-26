"""Tests for OpenCV fallback detectors."""

import numpy as np

from src.modules.face_detection.detectors.opencv_face_fallback import OpenCVFaceDetector
from src.modules.face_detection.detectors.opencv_person_fallback import OpenCVPersonDetector


def test_opencv_face_detector_on_blank_image(sample_rgb) -> None:
    detector = OpenCVFaceDetector()
    results = detector.detect(sample_rgb)
    assert isinstance(results, list)


def test_opencv_person_detector_on_blank_image(sample_rgb) -> None:
    detector = OpenCVPersonDetector()
    results = detector.detect(sample_rgb)
    assert isinstance(results, list)


def test_opencv_detectors_have_backend_name() -> None:
    assert OpenCVFaceDetector.backend_name == "opencv_haar"
    assert OpenCVPersonDetector.backend_name == "opencv_hog"


def test_opencv_face_detector_rejects_empty_array() -> None:
    detector = OpenCVFaceDetector()
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    results = detector.detect(empty)
    assert results == []
