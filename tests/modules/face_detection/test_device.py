"""Tests for device resolution."""

from src.modules.face_detection.device import resolve_device


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_string() -> None:
    device = resolve_device("auto")
    assert device in ("cpu", "cuda", "mps")
