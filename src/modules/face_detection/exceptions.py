"""Custom exceptions for Module 02 detection."""


class DetectionError(Exception):
    """Base exception for detection failures."""


class DetectorUnavailableError(DetectionError):
    """Raised when a detector backend cannot be loaded."""


class ImageValidationError(DetectionError):
    """Raised when input image data is invalid."""
