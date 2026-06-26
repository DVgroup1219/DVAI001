"""Configuration for Module 03 pixel analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PixelAnalysisConfig:
    """Runtime configuration for pixel analysis."""

    sample_size: int = 150_000
    dominant_cluster_count: int = 6
    output_dir: Path = field(default_factory=lambda: Path("cache/pixel_analysis"))
    json_dir_name: str = "analysis"
    csv_filename: str = "pixel_analysis.csv"
    summary_json_filename: str = "pixel_analysis_summary.json"

    def resolved_output_dir(self) -> Path:
        """Return absolute output directory."""
        return Path(self.output_dir).resolve()

    def json_dir(self) -> Path:
        """Per-image JSON output directory."""
        return self.resolved_output_dir() / self.json_dir_name
