"""Command-line interface for Module 03."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.modules.pixel_analysis.config import PixelAnalysisConfig
from src.modules.pixel_analysis.pipeline import PixelAnalysisPipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DV Studio Color AI — Module 03 — Pixel Analysis Engine",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a single JPEG image")
    group.add_argument("--folder", type=Path, help="Path to a folder of JPEG images")
    group.add_argument("--batch", nargs="+", type=Path, help="Multiple JPEG image paths")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cache/pixel_analysis"),
        help="Output directory for JSON and CSV",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    config = PixelAnalysisConfig(output_dir=args.output)
    pipeline = PixelAnalysisPipeline(config)

    try:
        if args.image:
            report = pipeline.run_single(args.image)
            if report.errors:
                for err in report.errors:
                    logger.error("Analysis error [%s]: %s", report.image, err)
            logger.info(
                "Done: %s — scene=%s green_cast=%s vegetation=%.1f%% sky=%.1f%%",
                report.image,
                report.scene_type.scene_type,
                report.color_cast.green_cast,
                report.vegetation.vegetation_percentage,
                report.sky.sky_percentage,
            )
        elif args.folder:
            reports = pipeline.run_folder(args.folder)
            logger.info("Done: processed %d images", len(reports))
        else:
            reports = pipeline.run_batch(args.batch)
            logger.info("Done: processed %d images", len(reports))
    except Exception as exc:
        logger.exception("Module 03 failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
