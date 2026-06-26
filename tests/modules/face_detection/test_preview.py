"""Tests for debug preview generator."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.modules.face_detection.models import (
    BoundingBox,
    DetectionReport,
    FaceDetection,
    PersonDetection,
)
from src.modules.face_detection.preview import PreviewGenerator


def test_preview_returns_none_on_write_failure(tmp_path: Path) -> None:
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    report = DetectionReport(image="test.jpg", people=0, face_count=0)
    output = tmp_path / "preview.jpg"
    with patch("src.modules.face_detection.preview.cv2.imwrite", return_value=False):
        result = PreviewGenerator().generate(rgb, report, output)
    assert result is None


def test_preview_saves_copy(tmp_path: Path) -> None:
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    report = DetectionReport(
        image="test.jpg",
        people=1,
        face_count=1,
        person_detections=[
            PersonDetection(
                id=1,
                confidence=0.9,
                bbox=BoundingBox(10, 10, 30, 50),
                position="top-left",
            )
        ],
        faces=[
            FaceDetection(
                id=1,
                confidence=0.95,
                bbox=BoundingBox(40, 20, 25, 30),
                size="small",
                orientation="front",
                position="center",
            )
        ],
    )
    output = tmp_path / "preview.jpg"
    with patch("src.modules.face_detection.preview.cv2.imwrite", return_value=True) as mock_write:
        result = PreviewGenerator().generate(rgb, report, output)
    assert result == output
    mock_write.assert_called_once()
