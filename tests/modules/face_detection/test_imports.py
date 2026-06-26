"""Verify all Module 02 imports resolve."""

import importlib


MODULE_IMPORTS = [
    "src.modules.face_detection",
    "src.modules.face_detection.batch",
    "src.modules.face_detection.cli",
    "src.modules.face_detection.config",
    "src.modules.face_detection.device",
    "src.modules.face_detection.engine",
    "src.modules.face_detection.estimators",
    "src.modules.face_detection.exceptions",
    "src.modules.face_detection.exporter",
    "src.modules.face_detection.loader",
    "src.modules.face_detection.models",
    "src.modules.face_detection.pipeline",
    "src.modules.face_detection.preview",
    "src.modules.face_detection.detectors",
    "src.modules.face_detection.detectors.availability",
    "src.modules.face_detection.detectors.base",
    "src.modules.face_detection.detectors.factory",
    "src.modules.face_detection.detectors.mediapipe_face",
    "src.modules.face_detection.detectors.opencv_face_fallback",
    "src.modules.face_detection.detectors.opencv_person_fallback",
    "src.modules.face_detection.detectors.yolo_person",
]


def test_all_module_imports_are_valid() -> None:
    for module_name in MODULE_IMPORTS:
        module = importlib.import_module(module_name)
        assert module is not None
