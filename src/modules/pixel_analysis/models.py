"""Data models for Module 03 pixel analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColorCastResult:
    """Color cast detection from pixel analysis."""

    green_cast: str = "None"
    green_cast_strength: float = 0.0
    magenta_cast: str = "None"
    magenta_cast_strength: float = 0.0
    blue_cast: str = "None"
    blue_cast_strength: float = 0.0
    yellow_cast: str = "None"
    yellow_cast_strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "green_cast": self.green_cast,
            "green_cast_strength": round(self.green_cast_strength, 2),
            "magenta_cast": self.magenta_cast,
            "magenta_cast_strength": round(self.magenta_cast_strength, 2),
            "blue_cast": self.blue_cast,
            "blue_cast_strength": round(self.blue_cast_strength, 2),
            "yellow_cast": self.yellow_cast,
            "yellow_cast_strength": round(self.yellow_cast_strength, 2),
        }


@dataclass
class SkinToneRegionResult:
    """Detected skin tone regions in the image."""

    skin_pixel_percentage: float = 0.0
    skin_region_count: int = 0
    average_skin_rgb: dict[str, float] = field(default_factory=dict)
    average_skin_lab: dict[str, float] = field(default_factory=dict)
    skin_tone_label: str = ""
    regions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skin_pixel_percentage": round(self.skin_pixel_percentage, 2),
            "skin_region_count": self.skin_region_count,
            "average_skin_rgb": self.average_skin_rgb,
            "average_skin_lab": self.average_skin_lab,
            "skin_tone_label": self.skin_tone_label,
            "regions": self.regions,
        }


@dataclass
class VegetationResult:
    """Vegetation coverage estimate."""

    vegetation_percentage: float = 0.0
    dominant_green_rgb: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vegetation_percentage": round(self.vegetation_percentage, 2),
            "dominant_green_rgb": self.dominant_green_rgb,
        }


@dataclass
class SkyResult:
    """Sky coverage estimate."""

    sky_percentage: float = 0.0
    dominant_sky_rgb: dict[str, float] = field(default_factory=dict)
    sky_detected_in_upper_region: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sky_percentage": round(self.sky_percentage, 2),
            "dominant_sky_rgb": self.dominant_sky_rgb,
            "sky_detected_in_upper_region": self.sky_detected_in_upper_region,
        }


@dataclass
class HistogramResult:
    """Overall RGB and luminance histograms."""

    rgb_red: list[int] = field(default_factory=lambda: [0] * 256)
    rgb_green: list[int] = field(default_factory=lambda: [0] * 256)
    rgb_blue: list[int] = field(default_factory=lambda: [0] * 256)
    luminance: list[int] = field(default_factory=lambda: [0] * 256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rgb_red": self.rgb_red,
            "rgb_green": self.rgb_green,
            "rgb_blue": self.rgb_blue,
            "luminance": self.luminance,
        }


@dataclass
class WhiteBalanceEstimate:
    """Estimated white balance from pixel statistics."""

    estimated_temperature_k: float = 0.0
    estimated_tint: str = ""
    tint_strength: float = 0.0
    method: str = "gray_world_lab"
    average_rgb: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_temperature_k": round(self.estimated_temperature_k, 1),
            "estimated_tint": self.estimated_tint,
            "tint_strength": round(self.tint_strength, 2),
            "method": self.method,
            "average_rgb": self.average_rgb,
        }


@dataclass
class DominantColorCluster:
    """One dominant color cluster."""

    name: str
    rgb: dict[str, float]
    percentage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rgb": self.rgb,
            "percentage": round(self.percentage, 2),
        }


@dataclass
class DominantColorResult:
    """Dominant color cluster analysis."""

    clusters: list[DominantColorCluster] = field(default_factory=list)
    primary_dominant_color: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusters": [c.to_dict() for c in self.clusters],
            "primary_dominant_color": self.primary_dominant_color,
        }


@dataclass
class SceneTypeResult:
    """Scene classification from pixel features."""

    scene_type: str = "Unknown"
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_type": self.scene_type,
            "confidence": round(self.confidence, 1),
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


@dataclass
class PixelAnalysisReport:
    """Complete pixel analysis for one JPEG image."""

    image: str
    image_path: str = ""
    width: int = 0
    height: int = 0
    color_cast: ColorCastResult = field(default_factory=ColorCastResult)
    skin_tone: SkinToneRegionResult = field(default_factory=SkinToneRegionResult)
    vegetation: VegetationResult = field(default_factory=VegetationResult)
    sky: SkyResult = field(default_factory=SkyResult)
    histogram: HistogramResult = field(default_factory=HistogramResult)
    white_balance: WhiteBalanceEstimate = field(default_factory=WhiteBalanceEstimate)
    dominant_colors: DominantColorResult = field(default_factory=DominantColorResult)
    scene_type: SceneTypeResult = field(default_factory=SceneTypeResult)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": "DV Studio Color AI - Module 03 - Pixel Analysis Engine",
            "image": self.image,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "color_cast": self.color_cast.to_dict(),
            "skin_tone": self.skin_tone.to_dict(),
            "vegetation": self.vegetation.to_dict(),
            "sky": self.sky.to_dict(),
            "histogram": self.histogram.to_dict(),
            "white_balance": self.white_balance.to_dict(),
            "dominant_colors": self.dominant_colors.to_dict(),
            "scene_type": self.scene_type.to_dict(),
            "errors": self.errors,
        }

    def summary_row(self) -> dict[str, Any]:
        """Flatten key fields for CSV export."""
        return {
            "file_name": self.image,
            "file_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "green_cast": self.color_cast.green_cast,
            "magenta_cast": self.color_cast.magenta_cast,
            "blue_cast": self.color_cast.blue_cast,
            "yellow_cast": self.color_cast.yellow_cast,
            "skin_pixel_percentage": self.skin_tone.skin_pixel_percentage,
            "skin_tone_label": self.skin_tone.skin_tone_label,
            "skin_region_count": self.skin_tone.skin_region_count,
            "vegetation_percentage": self.vegetation.vegetation_percentage,
            "sky_percentage": self.sky.sky_percentage,
            "primary_dominant_color": self.dominant_colors.primary_dominant_color,
            "estimated_temperature_k": self.white_balance.estimated_temperature_k,
            "estimated_tint": self.white_balance.estimated_tint,
            "scene_type": self.scene_type.scene_type,
            "scene_confidence": self.scene_type.confidence,
            "error_count": len(self.errors),
        }
