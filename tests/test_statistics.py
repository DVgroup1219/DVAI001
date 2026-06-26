"""Tests for statistics analyzer."""

import numpy as np

from analysis.statistics import StatisticsAnalyzer


def test_statistics_on_solid_color() -> None:
  rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
  analyzer = StatisticsAnalyzer()
  stats = analyzer.analyze(rgb)

  assert stats.brightness == 128.0
  assert stats.mean == 128.0
  assert 0 <= stats.highlight_percentage <= 100
  assert 0 <= stats.shadow_percentage <= 100
  assert stats.dynamic_range_estimate in ("Low", "Medium", "High")
  assert len(stats.histogram.red) == 256
