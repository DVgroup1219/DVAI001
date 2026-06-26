"""Tests for file scanner utilities."""

from pathlib import Path

from utils.file_scanner import FileScanner


def test_scan_folder_finds_jpegs(tmp_path: Path, sample_jpeg: Path) -> None:
  scanner = FileScanner()
  found = scanner.scan_folder(tmp_path)
  assert len(found) == 1
  assert found[0].name == "IMG001.JPG"


def test_deduplicate_removes_duplicates(sample_jpeg: Path) -> None:
  scanner = FileScanner()
  paths = [sample_jpeg, sample_jpeg]
  unique = scanner.deduplicate(paths)
  assert len(unique) == 1
