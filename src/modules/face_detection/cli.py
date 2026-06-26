"""Command-line interface for Module 02."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.modules.face_detection.config import DetectionConfig
from src.modules.face_detection.pipeline import FaceDetectionPipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="DV Studio Color AI — Module 02 — Face & Person Detection Engine",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a single JPEG image")
    group.add_argument("--folder", type=Path, help="Path to a folder of JPEG images")
    group.add_argument("--batch", nargs="+", type=Path, help="Multiple JPEG image paths")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cache/face_detection"),
        help="Output directory for JSON and previews",
    )
    parser.add_argument("--preview", action="store_true", help="Generate debug preview images")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device for YOLO person detection",
    )
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLOv8 model weights")
    parser.add_argument("--person-confidence", type=float, default=0.45)
    parser.add_argument("--face-confidence", type=float, default=0.5)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    config = DetectionConfig(
        yolo_model=args.yolo_model,
        person_confidence=args.person_confidence,
        face_confidence=args.face_confidence,
        device=args.device,
        preview_enabled=args.preview,
        output_dir=args.output,
    )
    pipeline = FaceDetectionPipeline(config)

    try:
        if args.image:
            report = pipeline.run_single(args.image, preview=args.preview)
            logger.info(
                "Done: %s — people=%d faces=%d",
                report.image,
                report.people,
                report.face_count,
            )
        elif args.folder:
            reports = pipeline.run_folder(args.folder, preview=args.preview)
            logger.info("Done: processed %d images", len(reports))
        else:
            reports = pipeline.run_batch(args.batch, preview=args.preview)
            logger.info("Done: processed %d images", len(reports))
    except Exception as exc:
        logger.exception("Module 02 failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
