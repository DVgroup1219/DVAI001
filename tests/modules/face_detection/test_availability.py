"""Tests for backend availability helpers."""

from src.modules.face_detection.detectors.availability import (
    is_mediapipe_available,
    is_torch_available,
    is_ultralytics_available,
)


def test_availability_helpers_return_bool() -> None:
    assert isinstance(is_ultralytics_available(), bool)
    assert isinstance(is_mediapipe_available(), bool)
    assert isinstance(is_torch_available(), bool)
