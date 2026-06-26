"""Configuration for Module 02 face and person detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    """Runtime configuration for detection engine."""

    yolo_model: str = "yolov8n.pt"
    person_class_id: int = 0
    person_confidence: float = 0.45
    face_confidence: float = 0.5
    face_model_selection: int = 1
    device: str = "auto"
    preview_enabled: bool = False
    output_dir: Path = field(default_factory=lambda: Path("cache/face_detection"))
    preview_dir_name: str = "previews"
    json_dir_name: str = "detections"

    def resolved_output_dir(self) -> Path:
        """Return absolute output directory path."""
        return Path(self.output_dir).resolve()

    def preview_dir(self) -> Path:
        """Directory for debug preview images."""
        return self.resolved_output_dir() / self.preview_dir_name

    def json_dir(self) -> Path:
        """Directory for per-image JSON results."""
        return self.resolved_output_dir() / self.json_dir_name
