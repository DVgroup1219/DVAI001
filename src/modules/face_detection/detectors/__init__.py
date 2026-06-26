"""Detector package for Module 02."""

from src.modules.face_detection.detectors.availability import (
    is_mediapipe_available,
    is_torch_available,
    is_ultralytics_available,
    log_backend_status,
)
from src.modules.face_detection.detectors.base import (
    BaseDetector,
    RawFaceDetection,
    RawPersonDetection,
)
from src.modules.face_detection.detectors.factory import create_face_detector, create_person_detector
from src.modules.face_detection.detectors.mediapipe_face import MediaPipeFaceDetector
from src.modules.face_detection.detectors.opencv_face_fallback import OpenCVFaceDetector
from src.modules.face_detection.detectors.opencv_person_fallback import OpenCVPersonDetector
from src.modules.face_detection.detectors.yolo_person import YOLOPersonDetector

__all__ = [
    "BaseDetector",
    "RawFaceDetection",
    "RawPersonDetection",
    "MediaPipeFaceDetector",
    "YOLOPersonDetector",
    "OpenCVFaceDetector",
    "OpenCVPersonDetector",
    "create_face_detector",
    "create_person_detector",
    "is_ultralytics_available",
    "is_mediapipe_available",
    "is_torch_available",
    "log_backend_status",
]
