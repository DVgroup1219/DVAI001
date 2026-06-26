"""Shared fixtures for Module 02 tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.modules.face_detection.detectors.base import RawFaceDetection, RawPersonDetection
from src.modules.face_detection.models import BoundingBox


@pytest.fixture
def sample_rgb() -> np.ndarray:
    """Synthetic RGB image array."""
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_jpeg(tmp_path: Path) -> Path:
    """Create a synthetic JPEG file."""
    rgb = np.zeros((200, 300, 3), dtype=np.uint8)
    rgb[:, :, 1] = 140
    path = tmp_path / "IMG001.JPG"
    Image.fromarray(rgb, mode="RGB").save(path, format="JPEG")
    return path


@pytest.fixture
def mock_person_detection() -> RawPersonDetection:
    """Mock person detection."""
    return RawPersonDetection(
        bbox=BoundingBox(x=50, y=40, width=100, height=180),
        confidence=0.92,
    )


@pytest.fixture
def mock_face_detection() -> RawFaceDetection:
    """Mock face detection with keypoints."""
    return RawFaceDetection(
        bbox=BoundingBox(x=120, y=60, width=80, height=90),
        confidence=0.99,
        keypoints={
            "right_eye": (140.0, 90.0),
            "left_eye": (170.0, 90.0),
            "nose_tip": (155.0, 105.0),
            "mouth_center": (155.0, 120.0),
        },
    )


@pytest.fixture
def mock_engine(sample_rgb, mock_person_detection, mock_face_detection):
    """Engine with mocked detectors."""
    from src.modules.face_detection.engine import FacePersonDetectionEngine

    engine = FacePersonDetectionEngine()
    person_detector = MagicMock()
    person_detector.detect.return_value = [mock_person_detection]
    face_detector = MagicMock()
    face_detector.detect.return_value = [mock_face_detection]
    engine._person_detector = person_detector
    engine._face_detector = face_detector
    return engine
