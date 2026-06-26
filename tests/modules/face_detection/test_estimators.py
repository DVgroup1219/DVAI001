"""Tests for face estimators."""

from src.modules.face_detection.detectors.base import RawFaceDetection
from src.modules.face_detection.estimators import (
    estimate_face_size,
    estimate_orientation,
    estimate_position,
)
from src.modules.face_detection.models import BoundingBox


def test_estimate_face_size() -> None:
    small = BoundingBox(0, 0, 50, 50)
    assert estimate_face_size(small, 1000, 1000) == "small"

    medium = BoundingBox(0, 0, 250, 250)
    assert estimate_face_size(medium, 1000, 1000) == "medium"

    large = BoundingBox(0, 0, 400, 400)
    assert estimate_face_size(large, 1000, 1000) == "large"


def test_estimate_position() -> None:
    center = BoundingBox(140, 110, 40, 40)
    assert estimate_position(center, 320, 240) == "center"

    top_left = BoundingBox(10, 10, 30, 30)
    assert estimate_position(top_left, 320, 240) == "top-left"


def test_estimate_orientation_front() -> None:
    face = RawFaceDetection(
        bbox=BoundingBox(0, 0, 80, 90),
        confidence=0.95,
        keypoints={
            "right_eye": (30.0, 40.0),
            "left_eye": (60.0, 40.0),
            "nose_tip": (45.0, 55.0),
        },
    )
    assert estimate_orientation(face) == "front"
