"""JSON export for detection results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.modules.face_detection.models import DetectionReport

logger = logging.getLogger(__name__)


def export_detection_json(report: DetectionReport, output_path: Path) -> Path:
    """Write a single image detection report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.error("Failed to write detection JSON %s: %s", output_path, exc, exc_info=True)
        raise
    logger.info("Exported detection JSON: %s", output_path)
    return output_path


def export_batch_summary(reports: list[DetectionReport], output_path: Path) -> Path:
    """Write batch detection summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "module": "DV Studio Color AI - Module 02 - Face & Person Detection",
        "image_count": len(reports),
        "total_people": sum(r.people for r in reports),
        "total_faces": sum(r.face_count for r in reports),
        "error_count": sum(1 for r in reports if r.errors),
        "images": [r.to_dict() for r in reports],
    }
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.error("Failed to write batch summary %s: %s", output_path, exc, exc_info=True)
        raise
    logger.info("Exported batch summary: %s", output_path)
    return output_path
