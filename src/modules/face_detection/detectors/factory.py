"""Detector factory with graceful backend fallback."""

from __future__ import annotations

import logging

from src.modules.face_detection.config import DetectionConfig
from src.modules.face_detection.detectors.base import BaseDetector, RawFaceDetection, RawPersonDetection
from src.modules.face_detection.detectors.mediapipe_face import MediaPipeFaceDetector
from src.modules.face_detection.detectors.opencv_face_fallback import OpenCVFaceDetector
from src.modules.face_detection.detectors.opencv_person_fallback import OpenCVPersonDetector
from src.modules.face_detection.detectors.yolo_person import YOLOPersonDetector
from src.modules.face_detection.exceptions import DetectorUnavailableError

logger = logging.getLogger(__name__)


def create_person_detector(
    config: DetectionConfig,
    device: str,
) -> BaseDetector[RawPersonDetection]:
    """
    Create a person detector, falling back to OpenCV HOG if YOLO is unavailable.

    Raises DetectorUnavailableError only when both backends fail.
    """
    try:
        detector = YOLOPersonDetector(
            model_name=config.yolo_model,
            confidence=config.person_confidence,
            person_class_id=config.person_class_id,
            device=device,
        )
        logger.info("Person detection backend: yolov8")
        return detector
    except (DetectorUnavailableError, RuntimeError, ImportError, OSError) as exc:
        logger.warning(
            "YOLO person detector unavailable (%s). Falling back to OpenCV HOG.",
            exc,
        )

    try:
        detector = OpenCVPersonDetector()
        logger.info("Person detection backend: opencv_hog (fallback)")
        return detector
    except Exception as exc:
        logger.error("All person detection backends failed: %s", exc, exc_info=True)
        raise DetectorUnavailableError(
            "No person detection backend available. "
            "Install ultralytics or ensure OpenCV HOG is functional."
        ) from exc


def create_face_detector(config: DetectionConfig) -> BaseDetector[RawFaceDetection]:
    """
    Create a face detector, falling back to OpenCV Haar if MediaPipe is unavailable.

    Raises DetectorUnavailableError only when both backends fail.
    """
    try:
        detector = MediaPipeFaceDetector(
            min_confidence=config.face_confidence,
            model_selection=config.face_model_selection,
        )
        logger.info("Face detection backend: mediapipe")
        return detector
    except (DetectorUnavailableError, RuntimeError, ImportError, OSError) as exc:
        logger.warning(
            "MediaPipe face detector unavailable (%s). Falling back to OpenCV Haar.",
            exc,
        )

    try:
        detector = OpenCVFaceDetector()
        logger.info("Face detection backend: opencv_haar (fallback)")
        return detector
    except Exception as exc:
        logger.error("All face detection backends failed: %s", exc, exc_info=True)
        raise DetectorUnavailableError(
            "No face detection backend available. "
            "Install mediapipe or ensure OpenCV Haar cascade is functional."
        ) from exc
