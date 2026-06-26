"""YOLOv8 person detector."""

from __future__ import annotations

import logging

import numpy as np

from src.modules.face_detection.detectors.base import BaseDetector, RawPersonDetection
from src.modules.face_detection.models import BoundingBox

logger = logging.getLogger(__name__)


class YOLOPersonDetector(BaseDetector[RawPersonDetection]):
    """Detects people using Ultralytics YOLOv8."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.45,
        person_class_id: int = 0,
        device: str = "cpu",
    ) -> None:
        self._confidence = confidence
        self._person_class_id = person_class_id
        self._device = device
        self._model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load YOLO model with error handling."""
        try:
            from ultralytics import YOLO

            model = YOLO(model_name)
            logger.info("Loaded YOLO model: %s on %s", model_name, self._device)
            return model
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for person detection. "
                "Install with: pip install ultralytics"
            ) from exc
        except Exception as exc:
            logger.exception("Failed to load YOLO model")
            raise RuntimeError(f"Cannot load YOLO model: {model_name}") from exc

    def detect(self, image_rgb: np.ndarray) -> list[RawPersonDetection]:
        """Detect all people in the image."""
        height, width = image_rgb.shape[:2]
        results: list[RawPersonDetection] = []

        try:
            predictions = self._model.predict(
                source=image_rgb,
                classes=[self._person_class_id],
                conf=self._confidence,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("YOLO person detection failed")
            raise RuntimeError("Person detection failed") from exc

        if not predictions:
            return results

        boxes = predictions[0].boxes
        if boxes is None:
            return results

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w == 0 or h == 0:
                continue
            results.append(
                RawPersonDetection(
                    bbox=BoundingBox(x=x1, y=y1, width=w, height=h),
                    confidence=conf,
                )
            )

        logger.debug("Detected %d people", len(results))
        return results
