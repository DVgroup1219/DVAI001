"""File scanning utilities for JPEG discovery."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileScanner:
    """Scans folders and deduplicates JPEG file paths."""

    JPEG_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

    def is_jpeg(self, path: Path) -> bool:
        """Return True if path is a JPEG file."""
        return path.is_file() and path.suffix in self.JPEG_EXTENSIONS

    def scan_folder(self, folder: Path) -> list[Path]:
        """Recursively scan a folder for JPEG files."""
        if not folder.is_dir():
            logger.warning("Not a directory: %s", folder)
            return []

        results: list[Path] = []
        for path in folder.rglob("*"):
            if self.is_jpeg(path):
                results.append(path.resolve())

        results.sort()
        logger.info("Found %d JPEG files in %s", len(results), folder)
        return results

    def deduplicate(self, paths: list[Path]) -> list[Path]:
        """Remove duplicate paths while preserving order."""
        seen: set[Path] = set()
        unique: list[Path] = []
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen and self.is_jpeg(resolved):
                seen.add(resolved)
                unique.append(resolved)
        return unique
