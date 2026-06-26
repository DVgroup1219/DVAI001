"""Tests for individual pixel analyzers."""

from src.modules.pixel_analysis.analyzers.color_cast import ColorCastAnalyzer
from src.modules.pixel_analysis.analyzers.dominant_colors import DominantColorAnalyzer
from src.modules.pixel_analysis.analyzers.histogram import HistogramAnalyzer
from src.modules.pixel_analysis.analyzers.scene_type import SceneTypeAnalyzer
from src.modules.pixel_analysis.analyzers.skin_tone import SkinToneAnalyzer
from src.modules.pixel_analysis.analyzers.sky import SkyAnalyzer
from src.modules.pixel_analysis.analyzers.vegetation import VegetationAnalyzer
from src.modules.pixel_analysis.analyzers.white_balance import WhiteBalanceAnalyzer
from src.modules.pixel_analysis.models import ColorCastResult, DominantColorResult, SkinToneRegionResult, SkyResult, VegetationResult


def test_color_cast_on_green_image(green_rgb) -> None:
    result = ColorCastAnalyzer().analyze(green_rgb)
    assert result.green_cast in ("None", "Low", "Medium", "High")
    assert result.green_cast_strength >= 0


def test_vegetation_on_green_image(green_rgb) -> None:
    result = VegetationAnalyzer().analyze(green_rgb)
    assert result.vegetation_percentage > 50


def test_sky_on_sample(sample_rgb) -> None:
    result = SkyAnalyzer().analyze(sample_rgb)
    assert 0 <= result.sky_percentage <= 100


def test_histogram_lengths(sample_rgb) -> None:
    result = HistogramAnalyzer().analyze(sample_rgb)
    assert len(result.rgb_red) == 256
    assert len(result.luminance) == 256


def test_white_balance_returns_temperature(sample_rgb) -> None:
    result = WhiteBalanceAnalyzer().analyze(sample_rgb)
    assert 2500 <= result.estimated_temperature_k <= 10000
    assert result.estimated_tint != ""


def test_dominant_colors(sample_rgb) -> None:
    result = DominantColorAnalyzer(cluster_count=3).analyze(sample_rgb)
    assert len(result.clusters) >= 1
    assert result.primary_dominant_color != ""


def test_skin_tone_analyzer(sample_rgb) -> None:
    result = SkinToneAnalyzer().analyze(sample_rgb)
    assert 0 <= result.skin_pixel_percentage <= 100


def test_scene_type_classifier(sample_rgb) -> None:
    classifier = SceneTypeAnalyzer()
    result = classifier.classify(
        sample_rgb,
        ColorCastResult(),
        SkinToneRegionResult(),
        VegetationResult(vegetation_percentage=20),
        SkyResult(sky_percentage=15),
        DominantColorResult(primary_dominant_color="Green"),
    )
    assert result.scene_type in (
        "Outdoor Portrait",
        "Wedding Hall",
        "Indoor Flash",
        "Reception Stage",
        "Studio",
    )
    assert 0 <= result.confidence <= 100
