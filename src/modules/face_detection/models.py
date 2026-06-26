"""Data models for face and person detection results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def as_list(self) -> list[int]:
        """Return bbox as [x, y, w, h]."""
        return [self.x, self.y, self.width, self.height]

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class FaceDetection:
    """Detected face with metadata."""

    id: int
    confidence: float
    bbox: BoundingBox
    size: str
    orientation: str
    position: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize face detection to dictionary."""
        return {
            "id": self.id,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.as_list(),
            "size": self.size,
            "orientation": self.orientation,
            "position": self.position,
        }


@dataclass
class PersonDetection:
    """Detected person with metadata."""

    id: int
    confidence: float
    bbox: BoundingBox
    position: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize person detection to dictionary."""
        return {
            "id": self.id,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.as_list(),
            "position": self.position,
        }


@dataclass
class DetectionReport:
    """Complete detection result for one image."""

    image: str
    image_path: str = ""
    people: int = 0
    face_count: int = 0
    person_detections: list[PersonDetection] = field(default_factory=list)
    faces: list[FaceDetection] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report matching Module 02 JSON contract."""
        return {
            "image": self.image,
            "image_path": self.image_path,
            "people": self.people,
            "face_count": self.face_count,
            "person_detections": [p.to_dict() for p in self.person_detections],
            "faces": [f.to_dict() for f in self.faces],
            "errors": self.errors,
        }

    def to_json_ready(self) -> dict[str, Any]:
        """Alias for JSON export."""
        return self.to_dict()
