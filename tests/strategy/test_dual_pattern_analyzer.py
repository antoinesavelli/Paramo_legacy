# =====================================================
# tests/test_dual_pattern_analyzer.py
# =====================================================

from __future__ import annotations

import copy
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config.config import TradingConfig, AIAnalyzerConfig
from strategy.patterns.ai_pattern_analyzer import AIAnalyzerError
from strategy.patterns.dual_pattern_analyzer import DualPatternAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bars(n: int = 40) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame."""
    import numpy as np
    rng = np.random.default_rng(42)
    prices = 10.0 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-03 09:30", periods=n, freq="1min"),
        "open": prices,
        "high": prices + 0.05,
        "low": prices - 0.05,
        "close": prices,
        "volume": rng.integers(1000, 5000, n),
    })


def _hc_result(valid: bool, strength: float = 55.0) -> dict:
    return {
        "valid": valid,
        "symbol": "TST",
        "pattern_strength": strength,
        "patterns_detected": ["step_up"] if valid else [],
        "pattern_count": 1 if valid else 0,
        "min_score_threshold": 15.0,
        "gap_percent": 20.0,
        "step_ups": {"detected": valid, "step_count": 2, "retention_rate": 50.0, "total_advance": 1.0, "strength": strength},
        "parabolic": {"detected": False, "strength": 0},
        "breakout": {"detected": False, "strength": 0},
        "volume": {"volume_trend": "increasing", "avg_volume": 2000, "recent_volume": 2500, "high_volume_correlation": 30, "strength": 50},
        "support_resistance": {"support": [], "resistance": [], "current_price": 11.0, "strength": 40},
        "timestamp": "2024-01-03T09:30:00",
        "is_premarket_analyzed": False,
    }


def _claude_result(valid: bool, strength: float = 60.0) -> dict:
    base = _hc_result(valid, strength)
    base["pattern_strength"] = strength
    return base


def _make_dual(
    mode: str = "both",
    consensus: str = "and",
    enabled_in_backtest: bool = False,
    is_backtest: bool = False,
) -> tuple[DualPatternAnalyzer, MagicMock, MagicMock]:
    cfg = replace(
        TradingConfig().ai_analyzer,
        MODE=mode,
        CONSENSUS=consensus,
        ENABLED_IN_BACKTEST=enabled_in_backtest,
    )
    config = replace(TradingConfig(), ai_analyzer=cfg)

    hc_mock = MagicMock()
    ai_mock = MagicMock()

    dual = DualPatternAnalyzer(
        hard_coded=hc_mock,
        ai=ai_mock,
        config=config,
        is_backtest=is_backtest,
    )
    return dual, hc_mock, ai_mock


BARS = _make_bars()


# ---------------------------------------------------------------------------
# Mode: hard_coded_only
# ---------------------------------------------------------------------------

class TestHardCodedOnlyMode:
    def test_only_hc_called(self):
        dual, hc, claude = _make_dual(mode="hard_coded_only")
        hc.analyze_pattern.return_value = _hc_result(True)

        result = dual.analyze_pattern("TST", BARS, gap_percent=20.0)

        hc.analyze_pattern.assert_called_once()
        claude.analyze_pattern.assert_not_called()
        assert result["valid"] is True

    def test_output_has_required_keys(self):
        dual, hc, _ = _make_dual(mode="hard_coded_only")
        hc.analyze_pattern.return_value = _hc_result(True)
        result = dual.analyze_pattern("TST", BARS, gap_percent=20.0)
        required = {"valid", "symbol", "pattern_strength", "patterns_detected",
                    "pattern_count", "min_score_threshold", "step_ups",
                    "parabolic", "breakout", "volume", "support_resistance",
                    "timestamp", "is_premarket_analyzed"}
        assert required <= result.keys()


# ---------------------------------------------------------------------------
# Mode: claude_only
# ---------------------------------------------------------------------------

class TestAIOnlyMode:
    def test_ai_result_returned(self):
        dual, hc, ai = _make_dual(mode="ai_only")
        ai.analyze_pattern.return_value = _claude_result(True)

        result = dual.analyze_pattern("TST", BARS, gap_percent=120.0)

        ai.analyze_pattern.assert_called_once()
        hc.analyze_pattern.assert_not_called()
        assert result["valid"] is True

    def test_fallback_to_hc_on_error(self):
        dual, hc, ai = _make_dual(mode="ai_only")
        ai.analyze_pattern.side_effect = AIAnalyzerError("timeout")
        hc.analyze_pattern.return_value = _hc_result(True)

        result = dual.analyze_pattern("TST", BARS, gap_percent=20.0)

        hc.analyze_pattern.assert_called_once()
        assert result["valid"] is True
        assert "ai_fallback" in result.get("meta", {})

    def test_backtest_guard_skips_ai(self):
        dual, hc, ai = _make_dual(
            mode="ai_only", enabled_in_backtest=False, is_backtest=True
        )
        hc.analyze_pattern.return_value = _hc_result(True)

        dual.analyze_pattern("TST", BARS, gap_percent=20.0)

        ai.analyze_pattern.assert_not_called()
        hc.analyze_pattern.assert_called_once()


# ---------------------------------------------------------------------------
# Mode: both — consensus rules
# ---------------------------------------------------------------------------

class TestBothModeConsensus:
    def _run(self, consensus, hc_valid, claude_valid, hc_strength=55.0, claude_strength=60.0):
        dual, hc, claude = _make_dual(mode="both", consensus=consensus)
        hc.analyze_pattern.return_value = _hc_result(hc_valid, hc_strength)
        claude.analyze_pattern.return_value = _claude_result(claude_valid, claude_strength)
        return dual.analyze_pattern("TST", BARS, gap_percent=20.0)

    # AND
    def test_and_both_valid(self):
        assert self._run("and", True, True)["valid"] is True

    def test_and_hc_invalid(self):
        assert self._run("and", False, True)["valid"] is False

    def test_and_claude_invalid(self):
        assert self._run("and", True, False)["valid"] is False

    def test_and_both_invalid(self):
        assert self._run("and", False, False)["valid"] is False

    # OR
    def test_or_either_valid(self):
        assert self._run("or", False, True)["valid"] is True
        assert self._run("or", True, False)["valid"] is True

    def test_or_both_invalid(self):
        assert self._run("or", False, False)["valid"] is False

    def test_or_picks_higher_strength_when_both_valid(self):
        result = self._run("or", True, True, hc_strength=55.0, claude_strength=80.0)
        assert result["pattern_strength"] == pytest.approx(80.0)

    # primary_hard_coded
    def test_primary_hc_gates_validity(self):
        result = self._run("primary_hard_coded", False, True)
        assert result["valid"] is False  # HC is the gate

    def test_primary_hc_stores_ai_in_meta(self):
        result = self._run("primary_hard_coded", True, True)
        assert "ai" in result["meta"]

    # primary_ai
    def test_primary_ai_gates_validity(self):
        result = self._run("primary_ai", True, False)
        assert result["valid"] is False  # AI is the gate

    def test_primary_ai_stores_hc_in_meta(self):
        result = self._run("primary_ai", True, True)
        assert "hard_coded" in result["meta"]

    # Both modes store both raw results in meta
    def test_meta_contains_both_results(self):
        for consensus in ("and", "or", "primary_hard_coded", "primary_ai"):
            result = self._run(consensus, True, True)
            assert "hard_coded" in result["meta"]
            assert "ai" in result["meta"]


# ---------------------------------------------------------------------------
# Backtest guard (mode=both)
# ---------------------------------------------------------------------------

class TestBacktestGuard:
    def test_ai_never_called_in_backtest(self):
        dual, hc, ai = _make_dual(
            mode="both", enabled_in_backtest=False, is_backtest=True
        )
        hc.analyze_pattern.return_value = _hc_result(True)

        dual.analyze_pattern("TST", BARS, gap_percent=20.0)

        ai.analyze_pattern.assert_not_called()

    def test_ai_called_when_enabled_in_backtest(self):
        dual, hc, ai = _make_dual(
            mode="both", enabled_in_backtest=True, is_backtest=True
        )
        hc.analyze_pattern.return_value = _hc_result(True)
        ai.analyze_pattern.return_value = _claude_result(True)

        dual.analyze_pattern("TST", BARS, gap_percent=20.0)

        ai.analyze_pattern.assert_called_once()


# ---------------------------------------------------------------------------
# Fallback (mode=both, Claude raises)
# ---------------------------------------------------------------------------

class TestFallbackOnError:
    def test_returns_hc_result_on_ai_error(self):
        dual, hc, ai = _make_dual(mode="both", consensus="and")
        hc.analyze_pattern.return_value = _hc_result(True)
        ai.analyze_pattern.side_effect = AIAnalyzerError("API down")

        result = dual.analyze_pattern("TST", BARS, gap_percent=20.0)

        assert result["valid"] is True
        assert result.get("meta", {}).get("ai_error") is True


# ---------------------------------------------------------------------------
# ClaudePatternAnalyzer output schema (mocked Anthropic client)
# ---------------------------------------------------------------------------

class TestAIPatternAnalyzerSchema:
    def _make_analyzer(self, mock_response_text: str):
        """Build an AIPatternAnalyzer with the requests library fully mocked."""
        from strategy.patterns.ai_pattern_analyzer import AIPatternAnalyzer

        config = TradingConfig()
        analyzer = AIPatternAnalyzer.__new__(AIPatternAnalyzer)
        analyzer.config = config
        analyzer.logger = MagicMock()
        analyzer._cfg = config.ai_analyzer

        # Mock requests: POST → response.json()["message"]["content"]
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": mock_response_text}}
        mock_requests = MagicMock()
        mock_requests.post.return_value = mock_response
        mock_requests.Timeout = ConnectionError   # reuse a real exception class
        mock_requests.RequestException = OSError
        analyzer._requests = mock_requests
        return analyzer

    def test_all_required_keys_present(self):
        import json, datetime
        payload = {
            "valid": True,
            "symbol": "TST",
            "pattern_strength": 62.5,
            "patterns_detected": ["step_up"],
            "pattern_count": 1,
            "min_score_threshold": 15.0,
            "gap_percent": 20.0,
            "step_ups": {"detected": True, "step_count": 3, "retention_rate": 48.0,
                         "total_advance": 1.2, "strength": 65.0},
            "parabolic": {"detected": False, "angle": 0.0, "acceleration": 0.0,
                          "volume_multiplier": 0.0, "strength": 0.0, "angle_valid": False},
            "breakout": {"detected": False, "breakout_level": 0.0, "range_size": 0.0,
                         "volume_ratio": 0.0, "strength": 0.0},
            "volume": {"volume_trend": "increasing", "avg_volume": 2000.0,
                       "recent_volume": 2500.0, "high_volume_correlation": 35.0, "strength": 55.0},
            "support_resistance": {"support": [], "resistance": [], "current_price": 11.0,
                                   "strength": 40.0},
            "timestamp": datetime.datetime.now().isoformat(),
            "is_premarket_analyzed": False,
        }
        analyzer = self._make_analyzer(json.dumps(payload))
        result = analyzer.analyze_pattern("TST", _make_bars(), gap_percent=20.0)

        required = {"valid", "symbol", "pattern_strength", "patterns_detected",
                    "pattern_count", "min_score_threshold", "step_ups",
                    "parabolic", "breakout", "volume", "support_resistance",
                    "timestamp", "is_premarket_analyzed"}
        assert required <= result.keys()
        assert isinstance(result["valid"], bool)
        assert isinstance(result["pattern_strength"], float)

    def test_invalid_json_raises_ai_error(self):
        from strategy.patterns.ai_pattern_analyzer import AIAnalyzerError
        analyzer = self._make_analyzer("not json at all")
        with pytest.raises(AIAnalyzerError, match="non-JSON"):
            analyzer.analyze_pattern("TST", _make_bars(), gap_percent=20.0)

    def test_missing_keys_raises_ai_error(self):
        import json
        from strategy.patterns.ai_pattern_analyzer import AIAnalyzerError
        analyzer = self._make_analyzer(json.dumps({"valid": True}))
        with pytest.raises(AIAnalyzerError, match="missing keys"):
            analyzer.analyze_pattern("TST", _make_bars(), gap_percent=20.0)

    def test_think_tags_stripped_before_parse(self):
        """deepseek-r1 wraps reasoning in <think>...</think> — must be stripped."""
        import json, datetime
        payload = {
            "valid": True, "symbol": "TST", "pattern_strength": 50.0,
            "patterns_detected": [], "pattern_count": 0,
            "min_score_threshold": 40.0, "gap_percent": None,
            "step_ups": {"detected": False, "step_count": 0, "retention_rate": 0.0,
                         "total_advance": 0.0, "strength": 0.0},
            "parabolic": {"detected": False, "angle": 0.0, "acceleration": 0.0,
                          "volume_multiplier": 0.0, "strength": 0.0, "angle_valid": False},
            "breakout": {"detected": False, "breakout_level": 0.0, "range_size": 0.0,
                         "volume_ratio": 0.0, "strength": 0.0},
            "volume": {"volume_trend": "stable", "avg_volume": 0.0, "recent_volume": 0.0,
                       "high_volume_correlation": 0.0, "strength": 0.0},
            "support_resistance": {"support": [], "resistance": [], "current_price": 10.0,
                                   "strength": 0.0},
            "timestamp": datetime.datetime.now().isoformat(),
            "is_premarket_analyzed": False,
        }
        wrapped = f"<think>some reasoning here...</think>\n{json.dumps(payload)}"
        analyzer = self._make_analyzer(wrapped)
        result = analyzer.analyze_pattern("TST", _make_bars(), gap_percent=None)
        assert result["valid"] is True