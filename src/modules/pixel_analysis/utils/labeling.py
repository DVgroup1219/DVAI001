"""Label helpers for Module 03."""

from __future__ import annotations


def cast_level(value: float, low: float = 3.0, medium: float = 8.0) -> str:
    if value < low:
        return "None"
    if value < medium:
        return "Low"
    if value < 15.0:
        return "Medium"
    return "High"
