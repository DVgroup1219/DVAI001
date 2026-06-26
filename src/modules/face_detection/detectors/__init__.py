"""Detector package for Module 02."""

from src.modules.face_detection.detectors.base import BaseDetector, RawFaceDetection
from src.modules.face_detection.detectors.mediapipe_face import MediaPipeFaceDetector
from src.modules.face_detection.detectors.yolo_person import YOLOPersonDetector

__all__ = [
    "BaseDetector",
    "RawFaceDetection",
    "MediaPipeFaceDetector",
    "YOLOPersonDetector",
]
