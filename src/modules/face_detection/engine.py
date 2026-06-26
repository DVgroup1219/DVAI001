"""Main face and person detection engine."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.modules.face_detection.config import DetectionConfig
from src.modules.face_detection.detectors.mediapipe_face import MediaPipeFaceDetector
from src.modules.face_detection.detectors.yolo_person import YOLOPersonDetector
from src.modules.face_detection.device import resolve_device
from src.modules.face_detection.estimators import (
    estimate_face_size,
    estimate_orientation,
    estimate_position,
)
from src.modules.face_detection.loader import JPEGLoader, ImageLoadError
from src.modules.face_detection.models import (
    DetectionReport,
    FaceDetection,
    PersonDetection,
)

logger = logging.getLogger(__name__)


class FacePersonDetectionEngine:
    """
    Detects people and faces in Sony JPEG images.

    Read-only: never modifies source images or colors.
    Designed as a reusable foundation for Modules 03–06.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self._config = config or DetectionConfig()
        self._loader = JPEGLoader()
        self._device = resolve_device(self._config.device)
        self._person_detector: YOLOPersonDetector | None = None
        self._face_detector: MediaPipeFaceDetector | None = None

    @property
    def config(self) -> DetectionConfig:
        """Active detection configuration."""
        return self._config

    def _get_person_detector(self) -> YOLOPersonDetector:
        if self._person_detector is None:
            self._person_detector = YOLOPersonDetector(
                model_name=self._config.yolo_model,
                confidence=self._config.person_confidence,
                person_class_id=self._config.person_class_id,
                device=self._device,
            )
        return self._person_detector

    def _get_face_detector(self) -> MediaPipeFaceDetector:
        if self._face_detector is None:
            self._face_detector = MediaPipeFaceDetector(
                min_confidence=self._config.face_confidence,
                model_selection=self._config.face_model_selection,
            )
        return self._face_detector

    def detect_array(self, image_rgb: np.ndarray, image_name: str = "image.jpg") -> DetectionReport:
        """Run detection on an in-memory RGB image."""
        report = DetectionReport(image=image_name, image_path="")
        height, width = image_rgb.shape[:2]

        try:
            raw_people = self._get_person_detector().detect(image_rgb)
            raw_faces = self._get_face_detector().detect(image_rgb)
        except RuntimeError as exc:
            report.errors.append(str(exc))
            logger.error("Detection failed for %s: %s", image_name, exc)
            return report

        report.people = len(raw_people)
        report.face_count = len(raw_faces)

        for idx, person in enumerate(raw_people, start=1):
            report.person_detections.append(
                PersonDetection(
                    id=idx,
                    confidence=person.confidence,
                    bbox=person.bbox,
                    position=estimate_position(person.bbox, width, height),
                )
            )

        for idx, face in enumerate(raw_faces, start=1):
            report.faces.append(
                FaceDetection(
                    id=idx,
                    confidence=face.confidence,
                    bbox=face.bbox,
                    size=estimate_face_size(face.bbox, width, height),
                    orientation=estimate_orientation(face),
                    position=estimate_position(face.bbox, width, height),
                )
            )

        logger.info(
            "Detected %d people and %d faces in %s",
            report.people,
            report.face_count,
            image_name,
        )
        return report

    def detect_file(self, image_path: Path) -> DetectionReport:
        """Load a JPEG and run detection."""
        try:
            loaded = self._loader.load(image_path)
        except ImageLoadError as exc:
            report = DetectionReport(
                image=image_path.name,
                image_path=str(image_path.resolve()),
                errors=[str(exc)],
            )
            logger.error("Cannot load %s: %s", image_path, exc)
            return report

        report = self.detect_array(loaded.rgb, image_name=loaded.path.name)
        report.image_path = str(loaded.path)
        return report

    def close(self) -> None:
        """Release detector resources."""
        if self._face_detector is not None:
            self._face_detector.close()
            self._face_detector = None
        self._person_detector = None

    def __enter__(self) -> FacePersonDetectionEngine:
        return self

    def __exit__(self, *args) -> None:
        self.close()
