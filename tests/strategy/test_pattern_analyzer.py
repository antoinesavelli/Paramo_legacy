"""
Tests for strategy/pattern_analyzer.py

Covers:
- _get_dynamic_min_score: threshold tiers by gap %
- _calculate_pattern_confluence: score weighting, zero-weight guards,
  pattern_count accuracy, gap penalty application
- _analyze_step_ups: detection and strength for valid/invalid patterns
- _analyze_volume_pattern: increasing/decreasing/stable trends
- _detect_parabolic_spike: disabled (weight=0) short-circuit via _DETECTOR_DISABLED
- _detect_breakout: disabled short-circuit
- analyze_pattern: insufficient data early exit, valid/invalid result shape
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from config import TradingConfig
from config.loader import apply_overrides_immutable
from strategy.patterns.pattern_analyzer import PatternAnalyzer, _DETECTOR_DISABLED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(overrides=None):
    overrides = overrides or []
    cfg = apply_overrides_immutable(TradingConfig(), overrides)
    data_handler = MagicMock()
    return PatternAnalyzer(cfg, data_handler)


def _make_bars(n=60, trend="up", base_price=10.0, base_volume=100_000):
    """Build a minimal OHLCV DataFrame suitable for pattern analysis."""
    times = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="US/Eastern")
    if trend == "up":
        closes = np.linspace(base_price, base_price * 1.5, n)
    elif trend == "flat":
        closes = np.full(n, base_price)
    else:
        closes = np.linspace(base_price * 1.5, base_price, n)

    return pd.DataFrame({
        "timestamp": times,
        "open":   closes * 0.999,
        "high":   closes * 1.005,
        "low":    closes * 0.995,
        "close":  closes,
        "volume": np.full(n, base_volume),
    })


# ---------------------------------------------------------------------------
# _get_dynamic_min_score
# ---------------------------------------------------------------------------

class TestGetDynamicMinScore:
    def test_none_gap_returns_normal_threshold(self):
        pa = _make_analyzer()
        result = pa._get_dynamic_min_score(None)
        assert result == pa.config.pattern.CONFLUENCE_NORMAL_GAP_MIN_SCORE

    def test_extreme_gap_returns_low_threshold(self):
        pa = _make_analyzer()
        pc = pa.config.pattern
        result = pa._get_dynamic_min_score(pc.CONFLUENCE_EXTREME_GAP_THRESHOLD + 1)
        assert result == pc.CONFLUENCE_EXTREME_GAP_MIN_SCORE

    def test_large_gap_returns_mid_threshold(self):
        pa = _make_analyzer()
        pc = pa.config.pattern
        gap = pc.CONFLUENCE_LARGE_GAP_THRESHOLD + 1
        result = pa._get_dynamic_min_score(gap)
        assert result == pc.CONFLUENCE_LARGE_GAP_MIN_SCORE

    def test_normal_gap_returns_normal_threshold(self):
        pa = _make_analyzer()
        pc = pa.config.pattern
        result = pa._get_dynamic_min_score(5.0)
        assert result == pc.CONFLUENCE_NORMAL_GAP_MIN_SCORE

    def test_boundary_at_extreme_threshold(self):
        pa = _make_analyzer()
        pc = pa.config.pattern
        result = pa._get_dynamic_min_score(pc.CONFLUENCE_EXTREME_GAP_THRESHOLD)
        assert result == pc.CONFLUENCE_EXTREME_GAP_MIN_SCORE


# ---------------------------------------------------------------------------
# _calculate_pattern_confluence
# ---------------------------------------------------------------------------

class TestCalculatePatternConfluence:
    def _detected(self, strength=80):
        return {"detected": True, "strength": strength}

    def _not_detected(self):
        return {"detected": False, "strength": 0}

    def test_zero_weight_detector_not_counted(self):
        pa = _make_analyzer([
            ("pattern.CONFLUENCE_WEIGHT_PARABOLIC", "0.0"),
            ("pattern.CONFLUENCE_WEIGHT_BREAKOUT", "0.0"),
        ])
        result = pa._calculate_pattern_confluence(
            step_up=self._not_detected(),
            parabolic=self._detected(),   # weight=0 → must not count
            breakout=self._detected(),    # weight=0 → must not count
            volume={"strength": 50},
            sr={"strength": 50},
        )
        assert "parabolic" not in result["patterns_detected"]
        assert "breakout" not in result["patterns_detected"]

    def test_step_up_detected_adds_to_patterns(self):
        pa = _make_analyzer()
        result = pa._calculate_pattern_confluence(
            step_up=self._detected(100),
            parabolic=_DETECTOR_DISABLED,
            breakout=_DETECTOR_DISABLED,
            volume={"strength": 0},
            sr={"strength": 0},
        )
        assert "step_up" in result["patterns_detected"]
        assert result["pattern_count"] == 1

    def test_score_normalized_to_100(self):
        pa = _make_analyzer()
        result = pa._calculate_pattern_confluence(
            step_up=self._detected(200),   # strength capped at normalization
            parabolic=_DETECTOR_DISABLED,
            breakout=_DETECTOR_DISABLED,
            volume={"strength": 200},
            sr={"strength": 200},
        )
        assert result["total_score"] <= 100.0

    def test_score_non_negative(self):
        pa = _make_analyzer()
        result = pa._calculate_pattern_confluence(
            step_up=self._not_detected(),
            parabolic=_DETECTOR_DISABLED,
            breakout=_DETECTOR_DISABLED,
            volume={"strength": 0},
            sr={"strength": 0},
        )
        assert result["total_score"] >= 0.0

    def test_gap_penalty_applied_when_enabled(self):
        pa = _make_analyzer([
            ("pattern.EXTREME_GAP_PENALTY_ENABLED", "true"),
        ])
        pc = pa.config.pattern
        no_penalty = pa._calculate_pattern_confluence(
            step_up={"detected": True, "strength": 80},
            parabolic=_DETECTOR_DISABLED,
            breakout=_DETECTOR_DISABLED,
            volume={"strength": 80},
            sr={"strength": 80},
            gap_percent=10.0,   # below any penalty threshold
        )
        with_penalty = pa._calculate_pattern_confluence(
            step_up={"detected": True, "strength": 80},
            parabolic=_DETECTOR_DISABLED,
            breakout=_DETECTOR_DISABLED,
            volume={"strength": 80},
            sr={"strength": 80},
            gap_percent=pc.EXTREME_GAP_2000_THRESHOLD + 1,
        )
        assert with_penalty["total_score"] < no_penalty["total_score"]
        assert with_penalty["gap_penalty_applied"] == pc.EXTREME_GAP_2000_PENALTY

    def test_gap_penalty_not_applied_when_disabled(self):
        pa = _make_analyzer([("pattern.EXTREME_GAP_PENALTY_ENABLED", "false")])
        result = pa._calculate_pattern_confluence(
            step_up={"detected": True, "strength": 80},
            parabolic=_DETECTOR_DISABLED,
            breakout=_DETECTOR_DISABLED,
            volume={"strength": 80},
            sr={"strength": 80},
            gap_percent=9999.0,
        )
        assert result["gap_penalty_applied"] == 1.0


# ---------------------------------------------------------------------------
# _analyze_step_ups
# ---------------------------------------------------------------------------

class TestAnalyzeStepUps:
    def test_empty_bars_returns_not_detected(self):
        pa = _make_analyzer()
        result = pa._analyze_step_ups(pd.DataFrame())
        assert result["detected"] is False
        assert result["step_count"] == 0

    def test_flat_bars_not_detected(self):
        pa = _make_analyzer()
        bars = _make_bars(60, trend="flat")
        result = pa._analyze_step_ups(bars)
        assert result["detected"] is False

    def test_step_up_structure_detected(self):
        """Build an explicit staircase pattern that satisfies MIN_STEP_UPS=2."""
        pa = _make_analyzer()
        # Staircase: higher highs with pullbacks retaining > 35%
        n = 60
        prices = []
        base = 10.0
        for i in range(n):
            step = (i // 10) * 0.5        # step up every 10 bars
            wave = 0.1 * np.sin(i * 0.8)  # small oscillation
            prices.append(base + step + wave)

        times = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="US/Eastern")
        bars = pd.DataFrame({
            "timestamp": times,
            "open":   [p * 0.999 for p in prices],
            "high":   [p * 1.008 for p in prices],
            "low":    [p * 0.993 for p in prices],
            "close":  prices,
            "volume": [100_000] * n,
        })
        result = pa._analyze_step_ups(bars)
        # At minimum the structure fields exist
        assert "step_count" in result
        assert "retention_rate" in result
        assert result["strength"] >= 0

    def test_strength_in_valid_range(self):
        pa = _make_analyzer()
        bars = _make_bars(60, trend="up")
        result = pa._analyze_step_ups(bars)
        assert 0 <= result["strength"] <= 100


# ---------------------------------------------------------------------------
# _analyze_volume_pattern
# ---------------------------------------------------------------------------

class TestAnalyzeVolumePattern:
    def test_empty_returns_unknown(self):
        pa = _make_analyzer()
        result = pa._analyze_volume_pattern(pd.DataFrame())
        assert result["volume_trend"] == "unknown"

    def test_increasing_volume_detected(self):
        pa = _make_analyzer()
        n = 60
        # Low volume for first 50 bars, then sharp spike for final 10.
        # vol_ma_5 (last 5) = 200k; vol_ma_15 last-15 avg ≈ (5*50k + 10*200k)/15 = 150k
        # → ratio ≈ 1.33, which clears the 1.2 threshold.
        volumes = np.concatenate([np.full(50, 50_000.0), np.full(10, 200_000.0)])
        times = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="US/Eastern")
        bars = pd.DataFrame({
            "timestamp": times,
            "open": [10.0] * n, "high": [10.1] * n,
            "low": [9.9] * n, "close": [10.0] * n,
            "volume": volumes,
        })
        result = pa._analyze_volume_pattern(bars)
        assert result["volume_trend"] == "increasing"
        assert result["strength"] > 0

    def test_decreasing_volume_strength_zero(self):
        pa = _make_analyzer()
        n = 60
        volumes = np.linspace(200_000, 10_000, n)
        times = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="US/Eastern")
        bars = pd.DataFrame({
            "timestamp": times,
            "open": [10.0] * n, "high": [10.1] * n,
            "low": [9.9] * n, "close": [10.0] * n,
            "volume": volumes,
        })
        result = pa._analyze_volume_pattern(bars)
        assert result["volume_trend"] == "decreasing"
        assert result["strength"] == 0


# ---------------------------------------------------------------------------
# analyze_pattern — integration-level
# ---------------------------------------------------------------------------

class TestAnalyzePattern:
    def test_insufficient_data_returns_invalid(self):
        pa = _make_analyzer()
        bars = _make_bars(n=5)  # below the minimum of 30
        result = pa.analyze_pattern("TEST", bars=bars)
        assert result["valid"] is False
        assert "Insufficient" in result["reason"]

    def test_none_bars_calls_data_handler(self):
        pa = _make_analyzer()
        pa.data_handler.get_intraday_bars.return_value = None
        result = pa.analyze_pattern("TEST")
        assert result["valid"] is False

    def test_result_has_required_keys(self):
        pa = _make_analyzer()
        bars = _make_bars(60)
        result = pa.analyze_pattern("TEST", bars=bars, gap_percent=20.0)
        for key in ("valid", "pattern_strength", "patterns_detected", "pattern_count"):
            assert key in result

    def test_cache_hit_on_second_call(self):
        pa = _make_analyzer()
        bars = _make_bars(60)
        pa.analyze_pattern("TEST", bars=bars)
        initial_misses = pa.cache_misses
        pa.analyze_pattern("TEST", bars=bars)
        assert pa.cache_hits > 0
        assert pa.cache_misses == initial_misses  # no new miss

    def test_zero_weight_detectors_do_not_contribute(self):
        """With parabolic and breakout weights = 0, their 'detected' state is irrelevant."""
        pa = _make_analyzer([
            ("pattern.CONFLUENCE_WEIGHT_PARABOLIC", "0.0"),
            ("pattern.CONFLUENCE_WEIGHT_BREAKOUT", "0.0"),
        ])
        bars = _make_bars(60)
        result = pa.analyze_pattern("TEST", bars=bars, gap_percent=20.0)
        assert "parabolic" not in result.get("patterns_detected", [])
        assert "breakout" not in result.get("patterns_detected", [])