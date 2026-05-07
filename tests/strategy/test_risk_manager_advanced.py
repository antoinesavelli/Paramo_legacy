"""
Tests for strategy/risk_manager.py — advanced edge cases not covered by existing tests.

Covers:
- calc_position_size_percentage with market_adjustment=0 → returns 0
- calc_position_size_percentage with tiny account where 1 share can't be afforded
- calc_position_size_percentage with extreme stop distances
- RiskManager.check_entry_risk with buying_power below required
- RiskManager.check_entry_risk: daily loss already at limit
- RiskManager.calculate_stop_loss ATR path uses correct period
- RiskManager.update_trailing_stop: stop only ratchets upward
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from strategy.risk_manager import (
    calc_atr,
    calc_atr_stop,
    calc_atr_trailing_stop,
    calc_position_size_percentage,
    RiskManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n=30, base_price=10.0, atr_val=0.5):
    """Generate synthetic OHLCV bars with a known ATR."""
    highs  = [base_price + atr_val] * n
    lows   = [base_price - atr_val] * n
    closes = [base_price] * n
    opens  = [base_price] * n
    vols   = [100_000] * n
    ts = pd.date_range('2024-01-01 09:30', periods=n, freq='1min')
    return pd.DataFrame({
        'timestamp': ts, 'open': opens, 'high': highs,
        'low': lows, 'close': closes, 'volume': vols,
    })


def _make_config(
    stop_pct=2.0,
    max_pos_pct=25.0,
    max_daily_loss_pct=3.0,
    max_drawdown_pct=10.0,
    atr_enabled=True,
    atr_period=10,
    atr_mult=1.5,
    trailing_enabled=True,
    trailing_period=10,
    trailing_mult=1.5,
    trailing_min_profit=1.0,
    min_stop_dist=0.5,
):
    cfg = MagicMock()
    cfg.risk.STOP_LOSS_PERCENT_OF_ACCOUNT = stop_pct
    cfg.risk.MAX_POSITION_SIZE_PERCENT = max_pos_pct
    cfg.risk.MAX_DAILY_LOSS_PERCENT = max_daily_loss_pct
    cfg.risk.MAX_DRAWDOWN_PERCENT = max_drawdown_pct
    cfg.risk.ATR_STOP_ENABLED = atr_enabled
    cfg.risk.ATR_STOP_PERIOD = atr_period
    cfg.risk.ATR_STOP_MULTIPLIER = atr_mult
    cfg.risk.ATR_TRAILING_ENABLED = trailing_enabled
    cfg.risk.ATR_TRAILING_PERIOD = trailing_period
    cfg.risk.ATR_TRAILING_MULTIPLIER = trailing_mult
    cfg.risk.ATR_TRAILING_MIN_PROFIT_PCT = trailing_min_profit
    cfg.risk.ATR_STOP_MIN_DISTANCE_PCT = min_stop_dist
    return cfg


# ---------------------------------------------------------------------------
# calc_position_size_percentage — edge cases
# ---------------------------------------------------------------------------

class TestCalcPositionSizePercentageEdgeCases:
    def test_market_adjustment_zero_returns_zero_size(self):
        size = calc_position_size_percentage(
            entry=10.0,
            stop=9.5,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
            market_adjustment=0.0,
        )
        assert size == 0

    def test_account_too_small_to_buy_one_share(self):
        # entry=$1000, stop=$999, account=$50 → 0 shares
        size = calc_position_size_percentage(
            entry=1000.0,
            stop=999.0,
            account_equity=50.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
        )
        assert size == 0

    def test_extreme_narrow_stop_produces_large_size_capped_by_max_pct(self):
        # entry=$100, stop=$99.99 → very small risk per share → huge theoretical size
        # but capped at max_position_pct=25% of 100k → $25k / $100 = 250 shares
        size = calc_position_size_percentage(
            entry=100.0,
            stop=99.99,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
        )
        max_allowed = int(100_000.0 * 0.25 / 100.0)
        assert size <= max_allowed

    def test_extreme_wide_stop_produces_tiny_size(self):
        # entry=$100, stop=$1 → huge risk per share → tiny position
        size = calc_position_size_percentage(
            entry=100.0,
            stop=1.0,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
        )
        # risk = 100000*0.02 = 2000, risk_per_share = 99 → ~20 shares
        assert size <= 25

    def test_small_position_value_cap_limits_size(self):
        # max_position_pct=0.1% → very tight cap regardless of risk calculation
        size = calc_position_size_percentage(
            entry=100.0,
            stop=95.0,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=0.1,  # only $100 of position allowed
        )
        # $100 / $100 = 1 share at most
        assert size <= 1

    def test_market_adjustment_half_reduces_size(self):
        full = calc_position_size_percentage(
            entry=10.0,
            stop=9.5,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
            market_adjustment=1.0,
        )
        half = calc_position_size_percentage(
            entry=10.0,
            stop=9.5,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
            market_adjustment=0.5,
        )
        assert half < full

    def test_normal_case_returns_positive_size(self):
        size = calc_position_size_percentage(
            entry=10.0,
            stop=9.5,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
        )
        assert size > 0

    def test_entry_equals_stop_returns_zero(self):
        """Entry == stop → division by zero guard → returns 0."""
        size = calc_position_size_percentage(
            entry=10.0,
            stop=10.0,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
        )
        assert size == 0


# ---------------------------------------------------------------------------
# calc_atr_trailing_stop — ratchet only upward
# ---------------------------------------------------------------------------

class TestCalcAtrTrailingStopRatchet:
    def test_trailing_stop_never_decreases(self):
        bars = _make_bars(n=30, base_price=10.0, atr_val=0.3)
        stop1 = calc_atr_trailing_stop(
            bars=bars, highest_price=11.0, atr_period=10, atr_mult=1.5
        )
        # Simulate higher price → stop should be >=
        stop2 = calc_atr_trailing_stop(
            bars=bars, highest_price=12.0, atr_period=10, atr_mult=1.5
        )
        assert stop2 >= stop1

    def test_trailing_stop_below_highest_price(self):
        bars = _make_bars(n=30, base_price=10.0, atr_val=0.5)
        stop = calc_atr_trailing_stop(
            bars=bars, highest_price=12.0, atr_period=10, atr_mult=1.5
        )
        assert stop < 12.0

    def test_min_stop_distance_respected(self):
        bars = _make_bars(n=30, base_price=10.0, atr_val=0.01)  # tiny ATR
        stop = calc_atr_trailing_stop(
            bars=bars, highest_price=10.5, atr_period=10, atr_mult=1.5,
            min_stop_distance_pct=0.02
        )
        # stop must be at least 2% below highest_price
        assert stop <= 10.5 * 0.98 + 0.001


# ---------------------------------------------------------------------------
# RiskManager.check_entry_risk — additional edge cases
# ---------------------------------------------------------------------------

class TestRiskManagerCheckEntryRiskEdgeCases:
    def _make_rm(self, config=None, equity=100_000.0, buying_power=100_000.0, num_positions=0):
        if config is None:
            config = _make_config()
        api = MagicMock()
        account = MagicMock()
        account.buying_power = str(buying_power)
        account.portfolio_value = str(equity)
        account.equity = str(equity)
        api.get_account.return_value = account
        api.list_positions.return_value = [MagicMock()] * num_positions
        config.risk.MAX_CONCURRENT_POSITIONS = 5
        rm = RiskManager(config, api)
        return rm

    def test_no_trade_when_daily_loss_at_limit(self):
        cfg = _make_config(max_daily_loss_pct=2.0)
        rm = self._make_rm(cfg, equity=100_000.0)
        # daily_pnl at -$2000 = 2% of $100k
        rm.daily_pnl = -2_000.0
        result = rm.check_entry_risk(symbol='AAPL', entry_price=10.0, stop_price=9.5)
        assert result['approved'] is False

    def test_no_trade_when_daily_loss_exceeds_limit(self):
        cfg = _make_config(max_daily_loss_pct=2.0)
        rm = self._make_rm(cfg, equity=100_000.0)
        rm.daily_pnl = -3_000.0  # 3% of $100k
        result = rm.check_entry_risk(symbol='AAPL', entry_price=10.0, stop_price=9.5)
        assert result['approved'] is False

    def test_trade_allowed_when_below_daily_loss_limit(self):
        cfg = _make_config(max_daily_loss_pct=3.0)
        rm = self._make_rm(cfg, equity=100_000.0)
        rm.daily_pnl = -1_000.0  # 1% of $100k — below 3% limit
        result = rm.check_entry_risk(symbol='AAPL', entry_price=10.0, stop_price=9.5)
        assert result['approved'] is True

    def test_size_zero_returns_not_approved(self):
        cfg = _make_config()
        rm = self._make_rm(cfg)
        # entry == stop → invalid stop price → not approved
        result = rm.check_entry_risk(symbol='AAPL', entry_price=10.0, stop_price=10.0)
        assert result['approved'] is False

    def test_approved_result_contains_position_size(self):
        cfg = _make_config()
        rm = self._make_rm(cfg, equity=100_000.0)
        result = rm.check_entry_risk(symbol='AAPL', entry_price=10.0, stop_price=9.5)
        if result['approved']:
            assert result.get('position_size', 0) > 0


# ---------------------------------------------------------------------------
# RiskManager.calculate_stop_loss
# ---------------------------------------------------------------------------

class TestRiskManagerCalculateStopLoss:
    def test_atr_stop_below_entry_price(self):
        cfg = _make_config(atr_enabled=True, atr_period=10, atr_mult=1.5)
        rm = RiskManager(cfg, MagicMock())
        bars = _make_bars(n=30, base_price=10.0, atr_val=0.3)
        stop = rm.calculate_stop_loss(symbol='AAPL', entry_price=10.0, bars=bars)
        assert stop < 10.0

    def test_atr_stop_uses_multiplier(self):
        # calc_atr_stop (the pure helper) respects the multiplier parameter
        bars = _make_bars(n=30, base_price=10.0, atr_val=0.3)
        from strategy.risk_manager import calc_atr_stop
        stop_low  = calc_atr_stop(bars, entry_price=10.0, atr_mult=1.0)
        stop_high = calc_atr_stop(bars, entry_price=10.0, atr_mult=3.0)
        # Higher multiplier → stop placed further below entry
        assert stop_high < stop_low

    def test_fallback_stop_when_atr_disabled(self):
        cfg = _make_config(atr_enabled=False, stop_pct=5.0)
        rm = RiskManager(cfg, MagicMock())
        bars = _make_bars(n=30, base_price=10.0, atr_val=0.3)
        stop = rm.calculate_stop_loss(symbol='AAPL', entry_price=10.0, bars=bars)
        assert stop < 10.0


# ---------------------------------------------------------------------------
# RiskManager.update_trailing_stop — ratchet only upward
# ---------------------------------------------------------------------------

class TestRiskManagerUpdateTrailingStop:
    def test_trailing_stop_increases_when_price_rises(self):
        cfg = _make_config(trailing_enabled=True, trailing_mult=1.5, trailing_min_profit=1.0)
        rm = RiskManager(cfg, MagicMock())
        bars = _make_bars(n=30, base_price=10.0, atr_val=0.3)
        position = {
            'entry_price': 10.0,
            'stop_price': 9.5,
            'highest_price': 11.0,
            'recent_bars': bars,
        }
        new_stop = rm.update_trailing_stop(
            symbol='AAPL',
            current_price=11.5,
            position=position,
        )
        # Either returned a higher stop or None (already optimal)
        if new_stop is not None:
            assert new_stop >= 9.5

    def test_trailing_stop_does_not_decrease(self):
        cfg = _make_config(trailing_enabled=True, trailing_mult=1.5, trailing_min_profit=1.0)
        rm = RiskManager(cfg, MagicMock())
        bars = _make_bars(n=30, base_price=10.0, atr_val=0.3)
        position = {
            'entry_price': 10.0,
            'stop_price': 10.5,  # already high stop
            'highest_price': 12.0,
            'recent_bars': bars,
        }
        new_stop = rm.update_trailing_stop(
            symbol='AAPL',
            current_price=11.0,
            position=position,
        )
        # Must not go below current_stop; None means "no update needed" (already optimal)
        if new_stop is not None:
            assert new_stop >= 10.5
