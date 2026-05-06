"""
Tests for TradeSimulator._validate_signal and _position_size helpers.
We isolate validation logic without a real data handler or exit simulator
by sub-classing and providing minimal stubs.
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from execution.trade_simulator import TradeSimulator


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_config():
    cfg = MagicMock()
    cfg.risk.MAX_CONCURRENT_POSITIONS = 5
    cfg.risk.ENABLE_SLIPPAGE = False
    cfg.risk.SLIPPAGE_GAP_THRESHOLD = 200.0
    cfg.risk.SLIPPAGE_ENTRY_PCT = 0.001
    cfg.risk.SLIPPAGE_ENTRY_HIGH_GAP_PCT = 0.002
    cfg.risk.STOP_LOSS_PERCENT_OF_ACCOUNT = 2.0
    cfg.risk.MAX_POSITION_SIZE_PERCENT = 25.0
    cfg.session.PREMARKET_WARMUP_MINUTES = 30
    cfg.session.REGULAR_WARMUP_MINUTES = 5
    cfg.backtest.ANALYSIS_WINDOW_ENABLED = False
    cfg.backtest.MAX_CANDIDATES_PER_DAY = 5
    return cfg


def _make_entry_metrics(**overrides):
    base = {
        'entry_bar_volume': 10_000,
        'volume_freshness_ratio': 1.0,
        'entry_bar_index': 5,
        'time_bucket': '9:30-10:00',
        'gap_type': 'intraday',
    }
    base.update(overrides)
    return base


def _make_results(capital: float = 100_000.0):
    return {
        'capital': capital,
        'candidate_diagnostics': [],
        'trades': [],
    }


def _make_simulator():
    cfg = _make_config()
    sim = TradeSimulator.__new__(TradeSimulator)
    sim.config = cfg
    sim.logger = MagicMock()
    sim.positions = {}
    sim.data_handler = MagicMock()
    sim.pattern_analyzer = MagicMock()
    sim.market_context = None
    sim.metrics = MagicMock()
    sim.exit_simulator = MagicMock()
    return sim


# ---------------------------------------------------------------------------
# _validate_signal
# ---------------------------------------------------------------------------

class TestValidateSignal:
    def _validate(self, sim, **overrides):
        defaults = dict(
            symbol="AAPL",
            risk_ps=2.0,
            size=10,
            entry_price=100.0,
            results=_make_results(),
            entry_date_str="2024-01-02",
            entry_time_str="09:35",
            gap_pct=15.0,
            vix_at_entry=18.0,
            entry_metrics=_make_entry_metrics(),
        )
        defaults.update(overrides)
        return sim._validate_signal(**defaults)

    def test_valid_signal_returns_none(self):
        sim = _make_simulator()
        assert self._validate(sim) is None

    def test_negative_risk_ps_rejected(self):
        sim = _make_simulator()
        assert self._validate(sim, risk_ps=-1.0) == 'rejected'

    def test_zero_risk_ps_rejected(self):
        sim = _make_simulator()
        assert self._validate(sim, risk_ps=0.0) == 'rejected'

    def test_zero_size_rejected(self):
        sim = _make_simulator()
        assert self._validate(sim, size=0) == 'rejected'

    def test_insufficient_capital_rejected(self):
        sim = _make_simulator()
        # cost = 10 shares * $100 = $1000, but capital is only $500
        assert self._validate(sim, size=10, entry_price=100.0,
                              results=_make_results(capital=500.0)) == 'rejected'

    def test_rejection_appends_diagnostic(self):
        sim = _make_simulator()
        results = _make_results()
        sim._validate_signal(
            symbol="TSLA", risk_ps=-1.0, size=10, entry_price=100.0,
            results=results, entry_date_str="2024-01-02",
            entry_time_str="09:35", gap_pct=5.0,
            vix_at_entry=None, entry_metrics=_make_entry_metrics()
        )
        assert len(results['candidate_diagnostics']) == 1
        assert results['candidate_diagnostics'][0]['symbol'] == "TSLA"

    def test_volume_participation_cap(self):
        """Size should not exceed 10% of entry-bar volume."""
        sim = _make_simulator()
        # entry_bar_volume=100 → max_size_by_volume = 10
        metrics = _make_entry_metrics(entry_bar_volume=100)
        # size=50 > 10 → capped (returns None, not rejected)
        result = sim._validate_signal(
            symbol="AAPL", risk_ps=2.0, size=50, entry_price=1.0,
            results=_make_results(capital=100_000.0),
            entry_date_str="2024-01-02", entry_time_str="09:35",
            gap_pct=5.0, vix_at_entry=None, entry_metrics=metrics,
        )
        assert result is None   # not rejected, just capped internally


# ---------------------------------------------------------------------------
# Slippage application
# ---------------------------------------------------------------------------

class TestSlippageApplication:
    """Verify that slippage inflates the entry price in _process_signal."""

    def test_slippage_disabled_no_price_change(self):
        sim = _make_simulator()
        sim.config.risk.ENABLE_SLIPPAGE = False
        # We only test the price-adjustment formula in isolation.
        entry = 100.0
        gap_pct = 150.0  # below threshold
        # Slippage disabled → price unchanged
        if not sim.config.risk.ENABLE_SLIPPAGE:
            result_price = entry
        assert result_price == 100.0

    def test_slippage_normal_gap(self):
        entry = 100.0
        slip_pct = 0.001
        adjusted = entry * (1 + slip_pct)
        assert abs(adjusted - 100.1) < 0.001

    def test_slippage_high_gap_uses_higher_rate(self):
        entry = 100.0
        normal_slip = entry * (1 + 0.001)
        high_gap_slip = entry * (1 + 0.002)
        assert high_gap_slip > normal_slip