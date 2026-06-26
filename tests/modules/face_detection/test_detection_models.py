"""Tests for detection data models."""

from src.modules.face_detection.models import (
    BoundingBox,
    DetectionReport,
    FaceDetection,
    PersonDetection,
)


def test_bounding_box_helpers() -> None:
    box = BoundingBox(x=10, y=20, width=100, height=50)
    assert box.as_list() == [10, 20, 100, 50]
    assert box.area == 5000
    assert box.center == (60.0, 45.0)


def test_detection_report_json_shape() -> None:
    report = DetectionReport(
        image="IMG001.JPG",
        people=1,
        face_count=1,
        person_detections=[
            PersonDetection(
                id=1,
                confidence=0.9,
                bbox=BoundingBox(0, 0, 10, 10),
                position="center",
            )
        ],
        faces=[
            FaceDetection(
                id=1,
                confidence=0.99,
                bbox=BoundingBox(0, 0, 10, 10),
                size="small",
                orientation="front",
                position="center",
            )
        ],
    )
    data = report.to_dict()
    assert data["image"] == "IMG001.JPG"
    assert data["people"] == 1
    assert data["face_count"] == 1
    assert data["faces"][0]["orientation"] == "front"
