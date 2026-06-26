"""Verify Module 03 imports."""

import importlib

MODULES = [
    "src.modules.pixel_analysis",
    "src.modules.pixel_analysis.engine",
    "src.modules.pixel_analysis.exporter",
    "src.modules.pixel_analysis.batch",
    "src.modules.pixel_analysis.pipeline",
    "src.modules.pixel_analysis.cli",
    "src.modules.pixel_analysis.analyzers",
    "src.modules.pixel_analysis.analyzers.color_cast",
    "src.modules.pixel_analysis.analyzers.skin_tone",
    "src.modules.pixel_analysis.analyzers.vegetation",
    "src.modules.pixel_analysis.analyzers.sky",
    "src.modules.pixel_analysis.analyzers.histogram",
    "src.modules.pixel_analysis.analyzers.white_balance",
    "src.modules.pixel_analysis.analyzers.dominant_colors",
    "src.modules.pixel_analysis.analyzers.scene_type",
]


def test_all_imports_valid() -> None:
    for name in MODULES:
        assert importlib.import_module(name) is not None
