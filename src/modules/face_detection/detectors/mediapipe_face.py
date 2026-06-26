"""MediaPipe face detector."""

from __future__ import annotations

import logging

import numpy as np

from src.modules.face_detection.detectors.base import BaseDetector, RawFaceDetection
from src.modules.face_detection.exceptions import DetectorUnavailableError
from src.modules.face_detection.models import BoundingBox

logger = logging.getLogger(__name__)

_KEYPOINT_NAMES = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth_center",
    "right_ear",
    "left_ear",
)


class MediaPipeFaceDetector(BaseDetector[RawFaceDetection]):
    """Detects faces using MediaPipe Face Detection."""

    backend_name = "mediapipe"

    def __init__(
        self,
        min_confidence: float = 0.5,
        model_selection: int = 1,
    ) -> None:
        self._min_confidence = min_confidence
        self._model_selection = model_selection
        self._detector = self._create_detector()

    def _create_detector(self):
        """Initialize MediaPipe face detection."""
        try:
            import mediapipe as mp
        except ImportError as exc:
            logger.error("mediapipe not installed — MediaPipe face detection unavailable")
            raise DetectorUnavailableError(
                "mediapipe is required for MediaPipe face detection. "
                "Install with: pip install mediapipe"
            ) from exc

        try:
            return mp.solutions.face_detection.FaceDetection(
                model_selection=self._model_selection,
                min_detection_confidence=self._min_confidence,
            )
        except Exception as exc:
            logger.exception("Failed to initialize MediaPipe face detector")
            raise DetectorUnavailableError("Cannot initialize MediaPipe face detector") from exc

    def detect(self, image_rgb: np.ndarray) -> list[RawFaceDetection]:
        """Detect all faces in the image."""
        if image_rgb.size == 0:
            logger.warning("MediaPipe received empty image array")
            return []

        height, width = image_rgb.shape[:2]
        results: list[RawFaceDetection] = []

        try:
            mp_results = self._detector.process(image_rgb)
        except Exception as exc:
            logger.exception("MediaPipe face detection inference failed")
            raise RuntimeError("MediaPipe face detection failed") from exc

        if not mp_results or not mp_results.detections:
            return results

        for detection in mp_results.detections:
            try:
                score = float(detection.score[0]) if detection.score else 0.0
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
            except (AttributeError, IndexError, TypeError) as exc:
                logger.warning("Skipping malformed MediaPipe detection: %s", exc)
                continue

            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            if w <= 0 or h <= 0:
                continue

            keypoints = self._extract_keypoints(detection, width, height)
            results.append(
                RawFaceDetection(
                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                    confidence=score,
                    keypoints=keypoints,
                )
            )

        logger.debug("MediaPipe detected %d faces", len(results))
        return results

    def _extract_keypoints(
        self, detection, width: int, height: int
    ) -> dict[str, tuple[float, float]] | None:
        """Extract named keypoints for orientation estimation."""
        if not detection.location_data.relative_keypoints:
            return None

        points: dict[str, tuple[float, float]] = {}
        for idx, kp in enumerate(detection.location_data.relative_keypoints):
            if idx >= len(_KEYPOINT_NAMES):
                break
            points[_KEYPOINT_NAMES[idx]] = (kp.x * width, kp.y * height)
        return points or None

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception as exc:
                logger.warning("Error closing MediaPipe detector: %s", exc)
            finally:
                self._detector = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
