"""EXIF and dimension metadata analysis."""

from __future__ import annotations

import logging
from fractions import Fraction

from PIL import ExifTags
from PIL.Image import Image as PILImage

from core.image_loader import LoadedImage
from core.models import ExifData, ImageDimensions

logger = logging.getLogger(__name__)


class MetadataAnalyzer:
    """Extracts image dimensions and EXIF metadata."""

    _TAG_MAP = {v: k for k, v in ExifTags.TAGS.items()}
    _GP_TAG_MAP = {v: k for k, v in ExifTags.GPSTAGS.items()} if hasattr(ExifTags, "GPSTAGS") else {}

    def analyze_dimensions(self, loaded: LoadedImage) -> ImageDimensions:
        """Compute width, height, orientation, aspect ratio, and file size."""
        width, height = loaded.pil_image.size
        orientation = "Portrait" if height > width else "Landscape"
        if width == height:
            orientation = "Square"

        ratio = self._format_aspect_ratio(width, height)

        return ImageDimensions(
            width=width,
            height=height,
            orientation=orientation,
            aspect_ratio=ratio,
            file_size_bytes=loaded.file_size_bytes,
        )

    def analyze_exif(self, loaded: LoadedImage) -> ExifData:
        """Extract EXIF fields from a loaded image."""
        exif = ExifData()
        try:
            raw_exif = loaded.pil_image.getexif()
            if not raw_exif:
                return exif

            exif_dict = self._flatten_exif(raw_exif)
            exif.raw_tags = {str(k): str(v) for k, v in exif_dict.items()}

            exif.camera_model = self._get_tag(exif_dict, "Model")
            exif.lens = self._get_lens(exif_dict)
            exif.iso = self._get_tag(exif_dict, "ISOSpeedRatings", "PhotographicSensitivity")
            exif.aperture = self._format_aperture(exif_dict)
            exif.shutter_speed = self._format_shutter(exif_dict)
            exif.focal_length = self._format_focal_length(exif_dict)
            exif.white_balance = self._get_tag(exif_dict, "WhiteBalance")
            exif.color_space = self._get_tag(exif_dict, "ColorSpace")
            exif.exposure_compensation = self._format_exposure_comp(exif_dict)
            exif.metering_mode = self._get_tag(exif_dict, "MeteringMode")
            exif.flash = self._get_tag(exif_dict, "Flash")
        except Exception as exc:
            logger.warning("EXIF extraction failed for %s: %s", loaded.path, exc)

        return exif

    def is_sony_camera(self, exif: ExifData) -> bool:
        """Return True if EXIF indicates a Sony camera."""
        make = exif.raw_tags.get("Make", "").lower()
        model = exif.camera_model.lower()
        return "sony" in make or "sony" in model or "ilce" in model or "dsc-" in model

    def _flatten_exif(self, raw_exif) -> dict[str, object]:
        """Flatten EXIF tags including nested EXIF IFD."""
        result: dict[str, object] = {}
        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag_name == "ExifOffset" and hasattr(raw_exif, "get_ifd"):
                try:
                    ifd = raw_exif.get_ifd(tag_id)
                    for sub_id, sub_val in ifd.items():
                        sub_name = ExifTags.TAGS.get(sub_id, str(sub_id))
                        result[sub_name] = sub_val
                except Exception:
                    pass
            result[tag_name] = value
        return result

    def _get_tag(self, exif: dict[str, object], *names: str) -> str:
        """Get first matching EXIF tag as string."""
        for name in names:
            if name in exif and exif[name] is not None:
                return self._format_value(exif[name])
        return ""

    def _get_lens(self, exif: dict[str, object]) -> str:
        """Extract lens model from common Sony/EXIF fields."""
        for key in ("LensModel", "LensSpecification", "LensInfo"):
            if key in exif:
                return self._format_value(exif[key])
        return ""

    def _format_aperture(self, exif: dict[str, object]) -> str:
        """Format aperture from FNumber or ApertureValue."""
        if "FNumber" in exif:
            return f"f/{self._format_value(exif['FNumber'])}"
        if "ApertureValue" in exif:
            return f"f/{self._format_value(exif['ApertureValue'])}"
        return ""

    def _format_shutter(self, exif: dict[str, object]) -> str:
        """Format shutter speed from ExposureTime."""
        if "ExposureTime" not in exif:
            return ""
        value = exif["ExposureTime"]
        try:
            if isinstance(value, tuple):
                frac = Fraction(value[0], value[1])
            else:
                frac = Fraction(value).limit_denominator(10000)
            if frac.numerator == 1:
                return f"1/{frac.denominator}s"
            return f"{float(frac):.4f}s"
        except Exception:
            return self._format_value(value)

    def _format_focal_length(self, exif: dict[str, object]) -> str:
        """Format focal length in millimeters."""
        if "FocalLength" not in exif:
            return ""
        value = exif["FocalLength"]
        try:
            if isinstance(value, tuple):
                mm = value[0] / value[1]
            else:
                mm = float(value)
            return f"{mm:.1f}mm"
        except Exception:
            return self._format_value(value)

    def _format_exposure_comp(self, exif: dict[str, object]) -> str:
        """Format exposure compensation in EV."""
        if "ExposureBiasValue" not in exif:
            return ""
        value = exif["ExposureBiasValue"]
        try:
            if isinstance(value, tuple):
                ev = value[0] / value[1]
            else:
                ev = float(value)
            sign = "+" if ev >= 0 else ""
            return f"{sign}{ev:.1f} EV"
        except Exception:
            return self._format_value(value)

    def _format_aspect_ratio(self, width: int, height: int) -> str:
        """Return simplified aspect ratio string."""
        from math import gcd

        divisor = gcd(width, height)
        return f"{width // divisor}:{height // divisor}"

    def _format_value(self, value: object) -> str:
        """Convert EXIF value to readable string."""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore").strip("\x00")
            except Exception:
                return str(value)
        if isinstance(value, tuple):
            if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                if value[1] != 0:
                    return f"{value[0] / value[1]:.2f}"
            return str(value)
        return str(value)
