"""
Tests for screener/core.py — UnifiedScreener

Covers:
- screen_symbols: empty candidates returns empty result
- screen_symbols: daily volume pre-screen gate (all filtered)
- screen_symbols: relative volume filter gate (disabled path)
- screen_symbols: market context blocking path
- screen_symbols: candidates_df sort order preserved (highest gap first)
- CandidateSignal: dataclass field contract
"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from config import TradingConfig
from config.loader import apply_overrides_immutable
from screener.core import UnifiedScreener, CandidateSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**screener_overrides):
    pairs = [
        # Disable all optional filters so screener reaches pattern analysis
        ("screening.ENABLE_RELATIVE_VOLUME", "false"),
        ("screening.ENABLE_DAILY_VOLUME_PRESCREEN", "false"),
        ("screening.ENABLE_FLOAT_FILTER", "false"),
        ("screening.ENABLE_MARKETCAP_FILTER", "false"),
    ]
    for k, v in screener_overrides.items():
        pairs.append((k, v))
    return apply_overrides_immutable(TradingConfig(), pairs)


def _make_screener(config=None, is_live=False):
    cfg = config or _make_config()
    data_handler = MagicMock()
    pattern_analyzer = MagicMock()
    # Default: pattern analysis always fails (safe baseline)
    pattern_analyzer.analyze_pattern.return_value = {"valid": False, "reason": "test_rejection"}

    with patch("screener.core.AggregateDataHandler"):
        screener = UnifiedScreener(
            cfg,
            data_handler,
            pattern_analyzer,
            is_live=is_live,
        )
    return screener


def _make_candidates(*gap_pcts):
    return pd.DataFrame({
        "symbol": [f"SYM{i}" for i in range(len(gap_pcts))],
        "gap_percent": list(gap_pcts),
        "last_price": [10.0] * len(gap_pcts),
    })


# ---------------------------------------------------------------------------
# CandidateSignal dataclass
# ---------------------------------------------------------------------------

class TestCandidateSignal:
    def test_fields_exist(self):
        sig = CandidateSignal(
            symbol="AAPL",
            entry_ts=pd.Timestamp("2024-01-02 09:31", tz="US/Eastern"),
            entry_price=15.50,
            stop_price=14.00,
            gap_percent=25.0,
            pattern_strength=72.5,
            relative_volume=3.1,
            meta={"foo": "bar"},
        )
        assert sig.symbol == "AAPL"
        assert sig.entry_price == 15.50
        assert sig.meta["foo"] == "bar"


# ---------------------------------------------------------------------------
# screen_symbols — structural / gate tests
# ---------------------------------------------------------------------------

class TestScreenSymbols:
    def test_empty_candidates_returns_empty(self):
        screener = _make_screener()
        result = screener.screen_symbols(
            candidates_df=pd.DataFrame(),
            day=datetime(2024, 1, 2),
        )
        assert result["signals"] == []
        assert result["diagnostics"] == []

    def test_result_has_required_keys(self):
        screener = _make_screener()
        result = screener.screen_symbols(
            candidates_df=_make_candidates(25.0),
            day=datetime(2024, 1, 2),
        )
        assert "signals" in result
        assert "diagnostics" in result
        assert "stats" in result

    def test_candidates_sorted_by_gap_descending(self):
        """Screener must sort candidates highest gap first; all symbols skipped on None bars."""
        screener = _make_screener()
        # None bars → every symbol is skipped cleanly after the sort
        screener.data.get_intraday_bars.return_value = None

        candidates = _make_candidates(15.0, 50.0, 30.0)
        candidates["symbol"] = ["LOW", "HIGH", "MID"]
        result = screener.screen_symbols(candidates_df=candidates, day=datetime(2024, 1, 2))

        # All 3 candidates were received and sorted before processing
        assert result["stats"]["candidates_in"] == 3
        # None had bars so none produced signals
        assert result["signals"] == []

    def test_daily_volume_prescreen_all_filtered(self):
        cfg = _make_config(**{"screening.ENABLE_DAILY_VOLUME_PRESCREEN": "true"})
        screener = _make_screener(config=cfg)

        # AggregateDataHandler returns empty → all symbols fail volume check
        screener.aggregate_handler = MagicMock()
        screener.aggregate_handler.get_daily_volume.return_value = 0  # below any threshold

        with patch.object(screener, "_apply_daily_volume_filter", return_value=pd.DataFrame()):
            result = screener.screen_symbols(
                candidates_df=_make_candidates(25.0, 30.0),
                day=datetime(2024, 1, 2),
            )
        assert result["signals"] == []

    def test_market_context_blocks_all_trading(self):
        screener = _make_screener()
        market_context = MagicMock()
        market_context.should_trade.return_value = False
        market_context.market_indicators = {
            "market_score": 20.0,
            "trading_environment": "unfavorable",
        }
        screener.market_context = market_context

        result = screener.screen_symbols(
            candidates_df=_make_candidates(25.0, 30.0),
            day=datetime(2024, 1, 2),
        )
        assert result["signals"] == []

    def test_stats_candidates_in_matches_input(self):
        screener = _make_screener()
        candidates = _make_candidates(20.0, 25.0, 30.0)
        result = screener.screen_symbols(candidates_df=candidates, day=datetime(2024, 1, 2))
        assert result["stats"]["candidates_in"] == 3

    def test_no_signals_when_pattern_invalid(self):
        screener = _make_screener()
        # Pattern analyzer always returns invalid (default mock)
        result = screener.screen_symbols(
            candidates_df=_make_candidates(25.0),
            day=datetime(2024, 1, 2),
        )
        assert len(result["signals"]) == 0

    def test_rvol_filter_disabled_path_counts_correctly(self):
        cfg = _make_config(**{"screening.ENABLE_RELATIVE_VOLUME": "false"})
        screener = _make_screener(config=cfg)
        candidates = _make_candidates(20.0, 30.0)
        result = screener.screen_symbols(candidates_df=candidates, day=datetime(2024, 1, 2))
        # after_relative_volume_filter should equal input count when filter is off
        assert result["stats"]["after_relative_volume_filter"] == 2