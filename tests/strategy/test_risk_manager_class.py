# =====================================================
# test_risk_manager_class.py - RiskManager class tests
# =====================================================

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock
from config.config import TradingConfig
from config.loader import apply_overrides_immutable
from strategy.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_account(equity: float = 50_000.0, buying_power: float = 50_000.0):
    acct = MagicMock()
    acct.equity = str(equity)
    acct.buying_power = str(buying_power)
    acct.portfolio_value = str(equity)
    return acct


def _make_risk_manager(
    equity: float = 50_000.0,
    buying_power: float = 50_000.0,
    num_positions: int = 0,
    overrides: list = None,
) -> RiskManager:
    api = MagicMock()
    api.get_account.return_value = _make_account(equity, buying_power)
    api.list_positions.return_value = [MagicMock() for _ in range(num_positions)]

    cfg = apply_overrides_immutable(TradingConfig(), overrides or [])
    rm = RiskManager(cfg, api)
    return rm


def _make_bars(n: int = 20, base_price: float = 50.0) -> pd.DataFrame:
    np.random.seed(0)
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open":  prices,
        "high":  prices + 0.5,
        "low":   prices - 0.5,
        "close": prices,
        "volume": np.ones(n) * 1000,
    })


# ---------------------------------------------------------------------------
# TestCheckEntryRisk
# ---------------------------------------------------------------------------

class TestCheckEntryRisk:
    def test_approved_basic(self):
        rm = _make_risk_manager(equity=50_000.0, buying_power=50_000.0)
        result = rm.check_entry_risk("AAPL", entry_price=100.0, stop_price=95.0)
        assert result["approved"] is True
        assert result["position_size"] > 0

    def test_returns_dict_with_required_keys_on_approval(self):
        rm = _make_risk_manager(equity=50_000.0, buying_power=50_000.0)
        result = rm.check_entry_risk("AAPL", entry_price=100.0, stop_price=95.0)
        required_keys = {"approved", "position_size", "risk_amount", "risk_percent",
                         "stop_price", "entry_price", "market_adjustment"}
        assert required_keys <= result.keys()

    def test_rejected_daily_loss_limit(self):
        rm = _make_risk_manager(equity=50_000.0)
        # Simulate having already lost more than the daily limit
        max_daily_loss = 50_000.0 * (rm.config.risk.MAX_DAILY_LOSS_PERCENT / 100.0)
        rm.daily_pnl = -(max_daily_loss + 1.0)

        result = rm.check_entry_risk("TSLA", entry_price=200.0, stop_price=190.0)
        assert result["approved"] is False
        assert "daily" in result["reason"].lower()

    def test_rejected_max_concurrent_positions(self):
        max_pos = 3
        rm = _make_risk_manager(
            equity=50_000.0,
            num_positions=max_pos,
            overrides=[("risk.MAX_CONCURRENT_POSITIONS", str(max_pos))],
        )
        result = rm.check_entry_risk("NVDA", entry_price=300.0, stop_price=290.0)
        assert result["approved"] is False
        assert "position" in result["reason"].lower()

    def test_rejected_invalid_stop_above_entry(self):
        rm = _make_risk_manager()
        result = rm.check_entry_risk("GOOG", entry_price=100.0, stop_price=105.0)
        assert result["approved"] is False
        assert "stop" in result["reason"].lower() or "invalid" in result["reason"].lower()

    def test_rejected_insufficient_buying_power(self):
        # Tiny buying power forces position_size down to 0
        rm = _make_risk_manager(equity=50_000.0, buying_power=1.0)
        result = rm.check_entry_risk("META", entry_price=300.0, stop_price=280.0)
        assert result["approved"] is False

    def test_rejected_max_drawdown(self):
        rm = _make_risk_manager(equity=50_000.0)
        # Set a peak far above current equity to simulate drawdown
        rm.peak_balance = 50_000.0 * 2  # drawdown = 50 %
        result = rm.check_entry_risk("SPY", entry_price=400.0, stop_price=390.0)
        assert result["approved"] is False
        assert "drawdown" in result["reason"].lower()

    def test_market_adjustment_reduces_position_size(self):
        rm_full   = _make_risk_manager(equity=50_000.0, buying_power=50_000.0)
        rm_scaled = _make_risk_manager(equity=50_000.0, buying_power=50_000.0)

        full   = rm_full.check_entry_risk("AAPL", 100.0, 90.0, market_adjustment=1.0)
        scaled = rm_scaled.check_entry_risk("AAPL", 100.0, 90.0, market_adjustment=0.5)

        assert full["approved"] is True
        assert scaled["approved"] is True
        assert scaled["position_size"] <= full["position_size"]


# ---------------------------------------------------------------------------
# TestCalculateStopLoss
# ---------------------------------------------------------------------------

class TestCalculateStopLoss:
    def test_stop_below_entry(self):
        rm = _make_risk_manager()
        bars = _make_bars(n=30, base_price=50.0)
        stop = rm.calculate_stop_loss("AAPL", entry_price=55.0, bars=bars)
        assert stop < 55.0, f"Expected stop < entry, got {stop}"

    def test_fallback_on_empty_bars(self):
        rm = _make_risk_manager()
        stop = rm.calculate_stop_loss("AAPL", entry_price=100.0, bars=pd.DataFrame())
        assert stop < 100.0, "Fallback stop must be below entry price"
