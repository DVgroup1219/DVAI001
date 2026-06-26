"""Data models for Sony JPEG analysis results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExifData:
    """EXIF metadata extracted from a JPEG."""

    camera_model: str = ""
    lens: str = ""
    iso: str = ""
    aperture: str = ""
    shutter_speed: str = ""
    focal_length: str = ""
    white_balance: str = ""
    color_space: str = ""
    exposure_compensation: str = ""
    metering_mode: str = ""
    flash: str = ""
    raw_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ImageDimensions:
    """Basic image dimension metadata."""

    width: int = 0
    height: int = 0
    orientation: str = "Landscape"
    aspect_ratio: str = ""
    file_size_bytes: int = 0


@dataclass
class HistogramData:
    """Histogram bins for a color channel or composite."""

    red: list[int] = field(default_factory=lambda: [0] * 256)
    green: list[int] = field(default_factory=lambda: [0] * 256)
    blue: list[int] = field(default_factory=lambda: [0] * 256)
    l_channel: list[int] = field(default_factory=lambda: [0] * 256)
    a_channel: list[int] = field(default_factory=lambda: [0] * 256)
    b_channel: list[int] = field(default_factory=lambda: [0] * 256)
    h_channel: list[int] = field(default_factory=lambda: [0] * 180)
    s_channel: list[int] = field(default_factory=lambda: [0] * 256)
    v_channel: list[int] = field(default_factory=lambda: [0] * 256)


@dataclass
class ImageStatistics:
    """Pixel-level statistical analysis."""

    average_rgb: dict[str, float] = field(default_factory=dict)
    average_lab: dict[str, float] = field(default_factory=dict)
    average_hsv: dict[str, float] = field(default_factory=dict)
    histogram: HistogramData = field(default_factory=HistogramData)
    mean: float = 0.0
    median: float = 0.0
    standard_deviation: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    dynamic_range_estimate: str = ""
    highlight_percentage: float = 0.0
    shadow_percentage: float = 0.0


@dataclass
class SceneAnalysis:
    """Heuristic scene classification estimates."""

    indoor: bool = False
    outdoor: bool = False
    reception: bool = False
    stage: bool = False
    portrait: bool = False
    nature: bool = False
    night: bool = False
    golden_hour: bool = False
    backlight: bool = False
    flash: bool = False
    confidence_score: float = 0.0
    primary_scene: str = ""


@dataclass
class FaceAnalysis:
    """Per-face analysis results."""

    bounding_box: dict[str, int] = field(default_factory=dict)
    face_size: int = 0
    skin_area: int = 0
    average_skin_rgb: dict[str, float] = field(default_factory=dict)
    average_skin_lab: dict[str, float] = field(default_factory=dict)
    average_skin_hsv: dict[str, float] = field(default_factory=dict)
    highlight_percentage: float = 0.0
    shadow_percentage: float = 0.0


@dataclass
class BackgroundAnalysis:
    """Background color composition estimates."""

    dominant_colors: list[dict[str, Any]] = field(default_factory=list)
    green_percentage: float = 0.0
    blue_percentage: float = 0.0
    white_percentage: float = 0.0
    black_percentage: float = 0.0
    wood_brown_percentage: float = 0.0
    dominant_color: str = ""


@dataclass
class ColorAnalysis:
    """Overall color cast and saturation estimates."""

    warm: float = 0.0
    cool: float = 0.0
    neutral: float = 0.0
    green_cast: str = "None"
    magenta_cast: str = "None"
    yellow_cast: str = "None"
    blue_cast: str = "None"
    overall_saturation: float = 0.0
    overall_vibrance: float = 0.0
    skin_tone_label: str = ""


@dataclass
class QualityAnalysis:
    """Image quality metric estimates."""

    sharpness: float = 0.0
    noise: float = 0.0
    blur: float = 0.0
    over_exposure: float = 0.0
    under_exposure: float = 0.0
    clipping: float = 0.0
    sharpness_label: str = ""
    noise_label: str = ""
    blur_label: str = ""


@dataclass
class ImageAnalysis:
    """Complete analysis for a single JPEG image."""

    file_path: str
    file_name: str
    is_sony_jpeg: bool = True
    dimensions: ImageDimensions = field(default_factory=ImageDimensions)
    exif: ExifData = field(default_factory=ExifData)
    statistics: ImageStatistics = field(default_factory=ImageStatistics)
    scene: SceneAnalysis = field(default_factory=SceneAnalysis)
    faces: list[FaceAnalysis] = field(default_factory=list)
    background: BackgroundAnalysis = field(default_factory=BackgroundAnalysis)
    color: ColorAnalysis = field(default_factory=ColorAnalysis)
    quality: QualityAnalysis = field(default_factory=QualityAnalysis)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert analysis to a JSON-serializable dictionary."""
        return asdict(self)

    def summary_row(self) -> dict[str, Any]:
        """Flatten key fields for CSV export."""
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "width": self.dimensions.width,
            "height": self.dimensions.height,
            "orientation": self.dimensions.orientation,
            "aspect_ratio": self.dimensions.aspect_ratio,
            "file_size_bytes": self.dimensions.file_size_bytes,
            "camera_model": self.exif.camera_model,
            "lens": self.exif.lens,
            "iso": self.exif.iso,
            "aperture": self.exif.aperture,
            "shutter_speed": self.exif.shutter_speed,
            "focal_length": self.exif.focal_length,
            "white_balance": self.exif.white_balance,
            "color_space": self.exif.color_space,
            "exposure_compensation": self.exif.exposure_compensation,
            "metering_mode": self.exif.metering_mode,
            "flash": self.exif.flash,
            "indoor": self.scene.indoor,
            "outdoor": self.scene.outdoor,
            "reception": self.scene.reception,
            "stage": self.scene.stage,
            "portrait": self.scene.portrait,
            "nature": self.scene.nature,
            "night": self.scene.night,
            "golden_hour": self.scene.golden_hour,
            "backlight": self.scene.backlight,
            "flash_scene": self.scene.flash,
            "scene_confidence": self.scene.confidence_score,
            "primary_scene": self.scene.primary_scene,
            "faces_count": len(self.faces),
            "skin_tone": self.color.skin_tone_label,
            "highlight_percentage": self.statistics.highlight_percentage,
            "shadow_percentage": self.statistics.shadow_percentage,
            "green_cast": self.color.green_cast,
            "magenta_cast": self.color.magenta_cast,
            "yellow_cast": self.color.yellow_cast,
            "blue_cast": self.color.blue_cast,
            "dominant_color": self.background.dominant_color,
            "average_brightness": self.statistics.brightness,
            "dynamic_range_estimate": self.statistics.dynamic_range_estimate,
            "overall_saturation": self.color.overall_saturation,
            "overall_vibrance": self.color.overall_vibrance,
            "sharpness": self.quality.sharpness_label,
            "noise": self.quality.noise_label,
            "blur": self.quality.blur_label,
            "over_exposure": self.quality.over_exposure,
            "under_exposure": self.quality.under_exposure,
            "clipping": self.quality.clipping,
            "green_percentage": self.background.green_percentage,
            "blue_percentage": self.background.blue_percentage,
            "white_percentage": self.background.white_percentage,
            "black_percentage": self.background.black_percentage,
            "wood_brown_percentage": self.background.wood_brown_percentage,
        }


@dataclass
class AnalysisReport:
    """Collection of analyses for one or more images."""

    images: list[ImageAnalysis] = field(default_factory=list)
    output_directory: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "module": "DV Studio Color AI - Module 01 - Sony JPEG Analyzer",
            "image_count": len(self.images),
            "output_directory": self.output_directory,
            "images": [img.to_dict() for img in self.images],
        }
