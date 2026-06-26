"""Pixel sampling utilities."""

from __future__ import annotations

import numpy as np


def sample_pixels(pixels: np.ndarray, max_samples: int, seed: int = 42) -> np.ndarray:
    """Randomly sample pixels for faster analysis."""
    if len(pixels) <= max_samples:
        return pixels
    indices = np.random.default_rng(seed).choice(len(pixels), max_samples, replace=False)
    return pixels[indices]
