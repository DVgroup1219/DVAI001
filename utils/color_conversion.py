"""Color space conversion helpers."""

from __future__ import annotations

import cv2
import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 array to OpenCV LAB (L: 0-255, a/b: 0-255 offset)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 array to HSV (H: 0-179, S/V: 0-255)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def lab_to_display_values(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split LAB array into L, a*, b* display-friendly channels."""
    l_ch = lab[:, :, 0].astype(np.float32)
    a_ch = lab[:, :, 1].astype(np.float32) - 128.0
    b_ch = lab[:, :, 2].astype(np.float32) - 128.0
    return l_ch, a_ch, b_ch


def average_color_dict(
    rgb_mean: tuple[float, float, float],
    lab_mean: tuple[float, float, float],
    hsv_mean: tuple[float, float, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Build average RGB, LAB, HSV dictionaries."""
    rgb = {"r": rgb_mean[0], "g": rgb_mean[1], "b": rgb_mean[2]}
    lab = {"l": lab_mean[0], "a": lab_mean[1], "b": lab_mean[2]}
    hsv = {"h": hsv_mean[0], "s": hsv_mean[1], "v": hsv_mean[2]}
    return rgb, lab, hsv
