"""Runtime availability checks for optional detector backends."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def is_ultralytics_available() -> bool:
    """Return True if ultralytics YOLO can be imported."""
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


def is_mediapipe_available() -> bool:
    """Return True if mediapipe can be imported."""
    try:
        import mediapipe  # noqa: F401

        return True
    except ImportError:
        return False


def is_torch_available() -> bool:
    """Return True if PyTorch can be imported."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def log_backend_status() -> None:
    """Log which optional backends are available at startup."""
    logger.info(
        "Detector backends — ultralytics=%s mediapipe=%s torch=%s",
        is_ultralytics_available(),
        is_mediapipe_available(),
        is_torch_available(),
    )
