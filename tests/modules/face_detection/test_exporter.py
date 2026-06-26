"""Tests for JSON exporter."""

import json
from pathlib import Path

from src.modules.face_detection.exporter import export_batch_summary, export_detection_json
from src.modules.face_detection.models import DetectionReport


def test_export_detection_json(tmp_path: Path) -> None:
    report = DetectionReport(image="IMG001.JPG", people=0, face_count=0)
    path = tmp_path / "IMG001_detection.json"
    export_detection_json(report, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["image"] == "IMG001.JPG"


def test_export_batch_summary(tmp_path: Path) -> None:
    reports = [DetectionReport(image="A.JPG"), DetectionReport(image="B.JPG")]
    path = tmp_path / "summary.json"
    export_batch_summary(reports, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["image_count"] == 2
