"""Export analysis results to JSON and CSV."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from core.models import AnalysisReport

logger = logging.getLogger(__name__)


def export_json(report: AnalysisReport, output_path: Path) -> None:
    """Write full analysis report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
    logger.info("Wrote JSON report: %s", output_path)


def export_csv(report: AnalysisReport, output_path: Path) -> None:
    """Write flattened per-image summary to CSV."""
    if not report.images:
        logger.warning("No images to export to CSV")
        return

    rows = [img.summary_row() for img in report.images]
    fieldnames = list(rows[0].keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote CSV report: %s", output_path)
