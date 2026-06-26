"""Custom exceptions for Module 03."""

from __future__ import annotations


class PixelAnalysisError(Exception):
    """Base exception for pixel analysis failures."""


class ImageLoadError(PixelAnalysisError):
    """Raised when a JPEG cannot be loaded."""


class ImageValidationError(PixelAnalysisError):
    """Raised when pixel data is invalid."""
