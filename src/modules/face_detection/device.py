"""Device selection with GPU preference and CPU fallback."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """
    Resolve compute device for YOLO inference.

    Returns 'cuda', 'mps', or 'cpu'. MediaPipe runs on CPU by default.
    """
    if requested not in ("auto", "cuda", "mps", "cpu"):
        logger.warning("Unknown device '%s', falling back to auto", requested)
        requested = "auto"

    if requested == "cpu":
        return "cpu"

    try:
        import torch

        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but unavailable, using CPU")
            return "cpu"

        if requested == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but unavailable, using CPU")
            return "cpu"

        if torch.cuda.is_available():
            logger.info("Using GPU (CUDA) for person detection")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using GPU (MPS) for person detection")
            return "mps"
    except ImportError:
        logger.warning("PyTorch not available, using CPU")

    logger.info("Using CPU for person detection")
    return "cpu"
