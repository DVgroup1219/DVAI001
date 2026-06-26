"""Analyzer interfaces for Module 03."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class BasePixelAnalyzer(ABC, Generic[T]):
    """Abstract analyzer contract for pixel-level analysis."""

    @abstractmethod
    def analyze(self, rgb: np.ndarray) -> T:
        """Analyze pixels without modifying the image."""
