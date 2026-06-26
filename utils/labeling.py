"""Label mapping helpers for analysis outputs."""

from __future__ import annotations


def cast_level(value: float, low: float = 3.0, medium: float = 8.0) -> str:
    """Map a cast magnitude to None/Low/Medium/High."""
    if value < low:
        return "None"
    if value < medium:
        return "Low"
    if value < 15.0:
        return "Medium"
    return "High"


def quality_label(value: float, low: float, high: float, invert: bool = False) -> str:
    """Map a numeric quality score to Low/Medium/High."""
    if invert:
        if value <= low:
            return "High"
        if value <= high:
            return "Medium"
        return "Low"

    if value <= low:
        return "Low"
    if value <= high:
        return "Medium"
    return "High"


def dynamic_range_label(std_dev: float, highlight: float, shadow: float) -> str:
    """Estimate dynamic range category from spread and tonal extremes."""
    spread = highlight + shadow
    if std_dev < 35 and spread < 15:
        return "Low"
    if std_dev < 55 and spread < 30:
        return "Medium"
    return "High"
