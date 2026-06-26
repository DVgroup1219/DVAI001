"""Debug preview generator — never overwrites original images."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.modules.face_detection.models import DetectionReport

logger = logging.getLogger(__name__)

_PERSON_COLOR = (0, 200, 0)
_FACE_COLOR = (0, 120, 255)
_TEXT_COLOR = (255, 255, 255)


class PreviewGenerator:
    """Draws detection overlays on a copy for debugging only."""

    def generate(
        self,
        image_rgb: np.ndarray,
        report: DetectionReport,
        output_path: Path,
    ) -> Path | None:
        """
        Save a debug preview with bounding boxes and confidence scores.

        The original image file is never modified.
        """
        if report.errors:
            logger.warning("Skipping preview due to errors: %s", report.image)
            return None

        canvas = image_rgb.copy()
        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        for person in report.person_detections:
            self._draw_box(
                bgr,
                person.bbox.as_list(),
                _PERSON_COLOR,
                f"Person {person.id} {person.confidence:.2f}",
            )

        for face in report.faces:
            self._draw_box(
                bgr,
                face.bbox.as_list(),
                _FACE_COLOR,
                f"Face {face.id} {face.confidence:.2f} {face.orientation}",
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), bgr)
        if not success:
            logger.error("Failed to write debug preview: %s", output_path)
            return None
        logger.info("Saved debug preview: %s", output_path)
        return output_path

    def _draw_box(
        self,
        image_bgr: np.ndarray,
        bbox: list[int],
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        """Draw labeled rectangle on image copy."""
        x, y, w, h = bbox
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image_bgr,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
