"""Main face and person detection engine."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.modules.face_detection.config import DetectionConfig
from src.modules.face_detection.detectors.availability import log_backend_status
from src.modules.face_detection.detectors.base import BaseDetector, RawFaceDetection, RawPersonDetection
from src.modules.face_detection.detectors.factory import create_face_detector, create_person_detector
from src.modules.face_detection.device import resolve_device
from src.modules.face_detection.estimators import (
    estimate_face_size,
    estimate_orientation,
    estimate_position,
)
from src.modules.face_detection.exceptions import DetectorUnavailableError, ImageValidationError
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
        self._person_detector: BaseDetector[RawPersonDetection] | None = None
        self._face_detector: BaseDetector[RawFaceDetection] | None = None
        log_backend_status()

    @property
    def config(self) -> DetectionConfig:
        """Active detection configuration."""
        return self._config

    def _get_person_detector(self) -> BaseDetector[RawPersonDetection]:
        if self._person_detector is None:
            self._person_detector = create_person_detector(self._config, self._device)
        return self._person_detector

    def _get_face_detector(self) -> BaseDetector[RawFaceDetection]:
        if self._face_detector is None:
            self._face_detector = create_face_detector(self._config)
        return self._face_detector

    def _validate_rgb(self, image_rgb: np.ndarray, image_name: str) -> None:
        """Validate input array before detection."""
        if not isinstance(image_rgb, np.ndarray):
            raise ImageValidationError(f"Expected numpy array for {image_name}")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ImageValidationError(
                f"Expected HxWx3 RGB array for {image_name}, got shape {image_rgb.shape}"
            )
        if image_rgb.size == 0:
            raise ImageValidationError(f"Empty image array for {image_name}")

    def detect_array(self, image_rgb: np.ndarray, image_name: str = "image.jpg") -> DetectionReport:
        """Run detection on an in-memory RGB image."""
        report = DetectionReport(image=image_name, image_path="")

        try:
            self._validate_rgb(image_rgb, image_name)
        except ImageValidationError as exc:
            report.errors.append(str(exc))
            logger.error("Image validation failed for %s: %s", image_name, exc)
            return report

        height, width = image_rgb.shape[:2]
        raw_people: list[RawPersonDetection] = []
        raw_faces: list[RawFaceDetection] = []

        try:
            raw_people = self._get_person_detector().detect(image_rgb)
        except DetectorUnavailableError as exc:
            report.errors.append(f"person_detection_unavailable: {exc}")
            logger.error("Person detector unavailable for %s: %s", image_name, exc)
        except Exception as exc:
            report.errors.append(f"person_detection_failed: {exc}")
            logger.error(
                "Person detection failed for %s: %s",
                image_name,
                exc,
                exc_info=True,
            )

        try:
            raw_faces = self._get_face_detector().detect(image_rgb)
        except DetectorUnavailableError as exc:
            report.errors.append(f"face_detection_unavailable: {exc}")
            logger.error("Face detector unavailable for %s: %s", image_name, exc)
        except Exception as exc:
            report.errors.append(f"face_detection_failed: {exc}")
            logger.error(
                "Face detection failed for %s: %s",
                image_name,
                exc,
                exc_info=True,
            )

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

        if report.errors:
            logger.warning(
                "Detection completed with errors for %s: %s",
                image_name,
                "; ".join(report.errors),
            )
        else:
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
                errors=[f"load_failed: {exc}"],
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
        if self._person_detector is not None:
            self._person_detector.close()
            self._person_detector = None

    def __enter__(self) -> FacePersonDetectionEngine:
        return self

    def __exit__(self, *args) -> None:
        self.close()
