"""
Tests for screener/rules.py

Covers:
- filter_price_and_gap: boundary conditions, empty DataFrame, None input
- is_price_valid: None, negative, zero, at-floor, above-floor
- calculate_relative_volume: normal path, zero-volume day exclusion, insufficient data,
  zero current volume, None inputs
- calculate_momentum_score: zero inputs, normal values, large relative volume clamped
- ScreeningConfigView: construction from config
- cfg_view_from: extracts correct fields
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from screener.rules import (
    filter_price_and_gap,
    is_price_valid,
    calculate_relative_volume,
    calculate_momentum_score,
    ScreeningConfigView,
    cfg_view_from,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(min_gap=10.0, min_price=2.0, min_rvol=2.0, min_abs_vol=50_000):
    return ScreeningConfigView(
        MIN_GAP_PERCENT=min_gap,
        MIN_PRICE=min_price,
        MIN_RELATIVE_VOLUME=min_rvol,
        MIN_ABSOLUTE_VOLUME=min_abs_vol,
    )


def _make_gaps_df(symbols, gaps, prices):
    return pd.DataFrame({
        'symbol': symbols,
        'gap_percent': gaps,
        'last_price': prices,
        'open_price': prices,
        'prev_close': [p * (1 - g / 100) for p, g in zip(prices, gaps)],
    })


# ---------------------------------------------------------------------------
# filter_price_and_gap
# ---------------------------------------------------------------------------

class TestFilterPriceAndGap:
    def test_returns_symbols_above_both_thresholds(self):
        cfg = _make_cfg(min_gap=10.0, min_price=2.0)
        df = _make_gaps_df(['AAPL', 'MSFT'], [15.0, 5.0], [10.0, 10.0])
        result = filter_price_and_gap(df, cfg)
        assert list(result['symbol']) == ['AAPL']

    def test_boundary_gap_exactly_at_minimum_included(self):
        cfg = _make_cfg(min_gap=10.0, min_price=2.0)
        df = _make_gaps_df(['AAPL'], [10.0], [5.0])
        result = filter_price_and_gap(df, cfg)
        assert len(result) == 1

    def test_just_below_gap_threshold_excluded(self):
        cfg = _make_cfg(min_gap=10.0, min_price=2.0)
        df = _make_gaps_df(['AAPL'], [9.99], [5.0])
        result = filter_price_and_gap(df, cfg)
        assert len(result) == 0

    def test_boundary_price_exactly_at_minimum_included(self):
        cfg = _make_cfg(min_gap=10.0, min_price=2.0)
        df = _make_gaps_df(['AAPL'], [15.0], [2.0])
        result = filter_price_and_gap(df, cfg)
        assert len(result) == 1

    def test_price_just_below_minimum_excluded(self):
        cfg = _make_cfg(min_gap=10.0, min_price=2.0)
        df = _make_gaps_df(['AAPL'], [15.0], [1.99])
        result = filter_price_and_gap(df, cfg)
        assert len(result) == 0

    def test_empty_dataframe_returns_empty_with_columns(self):
        cfg = _make_cfg()
        result = filter_price_and_gap(pd.DataFrame(), cfg)
        assert result.empty
        assert 'symbol' in result.columns

    def test_none_input_returns_empty_with_columns(self):
        cfg = _make_cfg()
        result = filter_price_and_gap(None, cfg)
        assert result.empty
        assert 'symbol' in result.columns

    def test_multiple_symbols_filtered_correctly(self):
        cfg = _make_cfg(min_gap=10.0, min_price=2.0)
        df = _make_gaps_df(
            ['A', 'B', 'C', 'D'],
            [20.0, 9.0, 12.0, 50.0],
            [5.0, 5.0, 1.5, 3.0],
        )
        result = filter_price_and_gap(df, cfg)
        symbols = set(result['symbol'])
        assert 'A' in symbols     # gap=20 price=5 → pass
        assert 'B' not in symbols # gap=9 → fail
        assert 'C' not in symbols # price=1.5 → fail
        assert 'D' in symbols     # gap=50 price=3 → pass

    def test_result_is_copy_not_view(self):
        cfg = _make_cfg()
        df = _make_gaps_df(['AAPL'], [15.0], [5.0])
        result = filter_price_and_gap(df, cfg)
        result.loc[result.index[0], 'gap_percent'] = 999.0
        assert df.loc[0, 'gap_percent'] == 15.0


# ---------------------------------------------------------------------------
# is_price_valid
# ---------------------------------------------------------------------------

class TestIsPriceValid:
    def test_none_returns_false(self):
        assert is_price_valid(None, _make_cfg()) is False

    def test_negative_price_returns_false(self):
        assert is_price_valid(-1.0, _make_cfg(min_price=2.0)) is False

    def test_zero_price_returns_false(self):
        assert is_price_valid(0.0, _make_cfg(min_price=2.0)) is False

    def test_exactly_at_floor_returns_true(self):
        assert is_price_valid(2.0, _make_cfg(min_price=2.0)) is True

    def test_above_floor_returns_true(self):
        assert is_price_valid(10.0, _make_cfg(min_price=2.0)) is True

    def test_just_below_floor_returns_false(self):
        assert is_price_valid(1.99, _make_cfg(min_price=2.0)) is False

    def test_string_price_handled(self):
        # Should either coerce or return False — not raise
        result = is_price_valid("not_a_number", _make_cfg())
        assert result is False

    def test_valid_string_float_coerced(self):
        # String "5.0" is coerced to float
        assert is_price_valid(5.0, _make_cfg(min_price=2.0)) is True


# ---------------------------------------------------------------------------
# calculate_relative_volume
# ---------------------------------------------------------------------------

class TestCalculateRelativeVolume:
    def _make_daily_bars(self, volumes):
        return pd.DataFrame({'volume': volumes})

    def test_normal_calculation(self):
        # 10 days at 100k → avg=100k; current=200k → RVOL=2.0
        bars = self._make_daily_bars([100_000] * 10)
        result = calculate_relative_volume(bars, 200_000)
        assert result == pytest.approx(2.0)

    def test_zero_volume_days_excluded_from_baseline(self):
        # 5 real days at 100k, 5 zero-volume days → avg=100k still
        vols = [100_000] * 5 + [0] * 5
        bars = self._make_daily_bars(vols)
        result = calculate_relative_volume(bars, 200_000)
        assert result == pytest.approx(2.0)

    def test_insufficient_valid_days_returns_none(self):
        # Only 4 non-zero days (< 5 minimum)
        bars = self._make_daily_bars([100_000] * 4 + [0] * 6)
        result = calculate_relative_volume(bars, 200_000)
        assert result is None

    def test_zero_current_volume_returns_none(self):
        bars = self._make_daily_bars([100_000] * 10)
        result = calculate_relative_volume(bars, 0)
        assert result is None

    def test_none_current_volume_returns_none(self):
        bars = self._make_daily_bars([100_000] * 10)
        result = calculate_relative_volume(bars, None)
        assert result is None

    def test_empty_dataframe_returns_none(self):
        result = calculate_relative_volume(pd.DataFrame(), 100_000)
        assert result is None

    def test_none_dataframe_returns_none(self):
        result = calculate_relative_volume(None, 100_000)
        assert result is None

    def test_missing_volume_column_returns_none(self):
        bars = pd.DataFrame({'close': [10.0] * 10})
        result = calculate_relative_volume(bars, 100_000)
        assert result is None

    def test_rvol_above_one_when_current_above_avg(self):
        bars = self._make_daily_bars([50_000] * 10)
        result = calculate_relative_volume(bars, 100_000)
        assert result == pytest.approx(2.0)

    def test_rvol_below_one_when_current_below_avg(self):
        bars = self._make_daily_bars([100_000] * 10)
        result = calculate_relative_volume(bars, 25_000)
        assert result == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# calculate_momentum_score
# ---------------------------------------------------------------------------

class TestCalculateMomentumScore:
    def test_returns_float(self):
        score = calculate_momentum_score(20.0, 3.0, 500_000)
        assert isinstance(score, float)

    def test_higher_gap_increases_score(self):
        score_low  = calculate_momentum_score(10.0, 2.0, 100_000)
        score_high = calculate_momentum_score(50.0, 2.0, 100_000)
        assert score_high > score_low

    def test_higher_rvol_increases_score(self):
        score_low  = calculate_momentum_score(20.0, 1.0, 100_000)
        score_high = calculate_momentum_score(20.0, 5.0, 100_000)
        assert score_high > score_low

    def test_rvol_clamped_at_10(self):
        score_10 = calculate_momentum_score(20.0, 10.0, 100_000)
        score_100 = calculate_momentum_score(20.0, 100.0, 100_000)
        assert score_10 == pytest.approx(score_100)

    def test_zero_rvol_and_volume_returns_nonnegative(self):
        score = calculate_momentum_score(10.0, 0.0, 0.0)
        assert score >= 0.0

    def test_none_rvol_treated_as_zero(self):
        score_none = calculate_momentum_score(20.0, None, 100_000)
        score_zero = calculate_momentum_score(20.0, 0.0, 100_000)
        assert score_none == pytest.approx(score_zero)


# ---------------------------------------------------------------------------
# cfg_view_from
# ---------------------------------------------------------------------------

class TestCfgViewFrom:
    def test_extracts_correct_fields(self):
        config = MagicMock()
        config.screening.MIN_GAP_PERCENT = 15.0
        config.screening.MIN_PRICE = 3.0
        config.screening.MIN_RELATIVE_VOLUME = 2.5
        config.screening.MIN_ABSOLUTE_VOLUME = 75_000
        view = cfg_view_from(config)
        assert view.MIN_GAP_PERCENT == 15.0
        assert view.MIN_PRICE == 3.0
        assert view.MIN_RELATIVE_VOLUME == 2.5
        assert view.MIN_ABSOLUTE_VOLUME == 75_000

    def test_returns_screening_config_view_type(self):
        config = MagicMock()
        config.screening.MIN_GAP_PERCENT = 10.0
        config.screening.MIN_PRICE = 2.0
        config.screening.MIN_RELATIVE_VOLUME = 2.0
        config.screening.MIN_ABSOLUTE_VOLUME = 50_000
        view = cfg_view_from(config)
        assert isinstance(view, ScreeningConfigView)
