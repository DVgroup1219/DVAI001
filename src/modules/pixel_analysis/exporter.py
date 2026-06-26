"""JSON and CSV export for Module 03."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from src.modules.pixel_analysis.models import PixelAnalysisReport

logger = logging.getLogger(__name__)


def export_analysis_json(report: PixelAnalysisReport, output_path: Path) -> Path:
    """Write per-image pixel analysis to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.error("Failed to write JSON %s: %s", output_path, exc, exc_info=True)
        raise
    logger.info("Exported pixel analysis JSON: %s", output_path)
    return output_path


def export_analysis_csv(reports: list[PixelAnalysisReport], output_path: Path) -> Path:
    """Write flattened pixel analysis summary to CSV."""
    if not reports:
        logger.warning("No reports to export to CSV")
        return output_path

    rows = [r.summary_row() for r in reports]
    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        logger.error("Failed to write CSV %s: %s", output_path, exc, exc_info=True)
        raise
    logger.info("Exported pixel analysis CSV: %s", output_path)
    return output_path


def export_summary_json(reports: list[PixelAnalysisReport], output_path: Path) -> Path:
    """Write batch summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "module": "DV Studio Color AI - Module 03 - Pixel Analysis Engine",
        "image_count": len(reports),
        "error_count": sum(1 for r in reports if r.errors),
        "images": [r.to_dict() for r in reports],
    }
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.error("Failed to write summary JSON %s: %s", output_path, exc, exc_info=True)
        raise
    logger.info("Exported pixel analysis summary: %s", output_path)
    return output_path
