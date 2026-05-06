"""
Tests for pure helper functions in strategy/risk_manager.py.
RiskManager class itself is not tested here because it requires a live API
object; the pure math helpers are fully deterministic and have no I/O.
"""

import math
import pytest
import pandas as pd
import numpy as np

from strategy.risk_manager import (
    calc_atr,
    calc_atr_stop,
    calc_atr_trailing_stop,
    calc_position_size_percentage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 20, volatility: float = 0.5) -> pd.DataFrame:
    """Return a minimal OHLC DataFrame with predictable ATR."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, volatility, n))
    high = close + abs(rng.normal(0, volatility, n))
    low = close - abs(rng.normal(0, volatility, n))
    return pd.DataFrame({"high": high, "low": low, "close": close})


def _flat_bars(n: int = 20, price: float = 100.0) -> pd.DataFrame:
    """Return bars where high==low==close so ATR is nearly zero after warmup."""
    return pd.DataFrame(
        {"high": [price] * n, "low": [price] * n, "close": [price] * n}
    )


# ---------------------------------------------------------------------------
# calc_atr
# ---------------------------------------------------------------------------

class TestCalcAtr:
    def test_returns_positive_for_volatile_bars(self):
        bars = _make_bars(30, volatility=1.0)
        atr = calc_atr(bars, period=14)
        assert atr > 0.0

    def test_returns_zero_for_empty(self):
        assert calc_atr(pd.DataFrame(), period=14) == 0.0

    def test_returns_zero_for_none(self):
        assert calc_atr(None, period=14) == 0.0

    def test_returns_zero_when_too_few_bars(self):
        bars = _make_bars(5, volatility=1.0)
        assert calc_atr(bars, period=14) == 0.0

    def test_flat_bars_approach_zero(self):
        bars = _flat_bars(30)
        atr = calc_atr(bars, period=14)
        assert atr < 0.01


# ---------------------------------------------------------------------------
# calc_atr_stop
# ---------------------------------------------------------------------------

class TestCalcAtrStop:
    def test_atr_stop_below_entry(self):
        bars = _make_bars(30)
        stop = calc_atr_stop(bars, entry_price=100.0)
        assert stop < 100.0

    def test_fallback_when_too_few_bars(self):
        bars = _make_bars(5)
        stop = calc_atr_stop(bars, entry_price=100.0, fallback_pct=0.03)
        # should land near 100 * 0.97 = 97.0
        assert abs(stop - 97.0) < 0.5

    def test_fallback_when_none(self):
        stop = calc_atr_stop(None, entry_price=50.0, fallback_pct=0.05)
        assert abs(stop - 47.5) < 0.1

    def test_rounded_to_four_decimals(self):
        bars = _make_bars(30)
        stop = calc_atr_stop(bars, entry_price=100.0)
        assert stop == round(stop, 4)

    def test_custom_multiplier_widens_stop(self):
        bars = _make_bars(30, volatility=2.0)
        stop_1x = calc_atr_stop(bars, entry_price=100.0, atr_mult=1.0)
        stop_3x = calc_atr_stop(bars, entry_price=100.0, atr_mult=3.0)
        assert stop_3x < stop_1x


# ---------------------------------------------------------------------------
# calc_atr_trailing_stop
# ---------------------------------------------------------------------------

class TestCalcAtrTrailingStop:
    def test_trailing_stop_below_highest(self):
        bars = _make_bars(30)
        ts = calc_atr_trailing_stop(bars, highest_price=110.0)
        assert ts < 110.0

    def test_fallback_when_empty(self):
        ts = calc_atr_trailing_stop(pd.DataFrame(), highest_price=100.0, min_stop_distance_pct=0.02)
        assert abs(ts - 98.0) < 0.1

    def test_minimum_distance_respected(self):
        # Flat bars → ATR ≈ 0; minimum distance should still apply.
        bars = _flat_bars(30, price=100.0)
        ts = calc_atr_trailing_stop(bars, highest_price=100.0, min_stop_distance_pct=0.01)
        assert ts <= 100.0 * 0.99 + 0.001

    def test_larger_multiplier_widens_distance(self):
        bars = _make_bars(30, volatility=2.0)
        ts_tight = calc_atr_trailing_stop(bars, highest_price=120.0, atr_mult=0.5)
        ts_wide = calc_atr_trailing_stop(bars, highest_price=120.0, atr_mult=3.0)
        assert ts_wide < ts_tight


# ---------------------------------------------------------------------------
# calc_position_size_percentage
# ---------------------------------------------------------------------------

class TestCalcPositionSizePercentage:
    def test_basic_sizing(self):
        # $100k account, 1% risk, $100 entry, $98 stop → risk $2/share
        # risk budget = $1000 → 500 shares; position value cap = $25k → 250 shares
        size = calc_position_size_percentage(
            entry=100.0, stop=98.0,
            account_equity=100_000.0,
            stop_loss_pct=1.0,
            max_position_pct=25.0,
        )
        assert size == 250

    def test_zero_when_stop_above_entry(self):
        size = calc_position_size_percentage(
            entry=100.0, stop=105.0,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
        )
        assert size == 0

    def test_market_adjustment_reduces_size(self):
        base = calc_position_size_percentage(
            entry=100.0, stop=98.0,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
            market_adjustment=1.0,
        )
        reduced = calc_position_size_percentage(
            entry=100.0, stop=98.0,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
            market_adjustment=0.5,
        )
        assert reduced < base

    def test_never_negative(self):
        size = calc_position_size_percentage(
            entry=0.01, stop=0.009,
            account_equity=500.0,
            stop_loss_pct=0.1,
            max_position_pct=1.0,
        )
        assert size >= 0

    def test_zero_entry_price_safe(self):
        size = calc_position_size_percentage(
            entry=0.0, stop=0.0,
            account_equity=100_000.0,
            stop_loss_pct=2.0,
            max_position_pct=25.0,
        )
        assert size == 0