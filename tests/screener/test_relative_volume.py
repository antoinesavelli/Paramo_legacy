"""
Tests for screener/helpers.py

Covers LiveRelativeVolumeCalculator and BacktestRelativeVolumeCalculator.

- LiveRelativeVolumeCalculator.calculate_batch:
    - normal path: API returns daily bars and quote → correct RVOL
    - no quote data → symbol skipped
    - empty daily bars → symbol skipped
    - fewer than 5 valid days → symbol skipped
    - API exception → symbol skipped, no crash
    - zero current volume → symbol skipped

- BacktestRelativeVolumeCalculator.calculate_batch:
    - uses avg_volume_10d when available → correct RVOL
    - falls back to avg_volume_20d when 10d missing
    - falls back to manual calculation when neither available
    - zero avg_volume → symbol skipped
    - exception → symbol skipped, no crash
"""

import pytest
import pandas as pd
from datetime import datetime, date
from unittest.mock import MagicMock, patch

from screener.helpers import (
    LiveRelativeVolumeCalculator,
    BacktestRelativeVolumeCalculator,
    DiagnosticCreator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_live_config(lookback=10, min_rvol=2.0):
    cfg = MagicMock()
    cfg.screening.RELATIVE_VOLUME_LOOKBACK_DAYS = lookback
    cfg.screening.MIN_RELATIVE_VOLUME = min_rvol
    return cfg


def _make_backtest_config(lookback=10, pm_enabled=True):
    cfg = MagicMock()
    cfg.screening.RELATIVE_VOLUME_LOOKBACK_DAYS = lookback
    cfg.screening.MIN_RELATIVE_VOLUME = 2.0
    cfg.session.PREMARKET_ENABLED = pm_enabled
    cfg.session.PREMARKET_START_ET = "04:00"
    cfg.session.PREMARKET_WARMUP_MINUTES = 30
    cfg.session.REGULAR_WARMUP_MINUTES = 15
    return cfg


def _make_daily_bar_df(volumes, n=None):
    if n is None:
        n = len(volumes)
    dates = pd.date_range('2023-12-01', periods=n)
    return pd.DataFrame({
        'date': dates,
        'volume': volumes,
        'close': [100.0] * n,
    })


# ---------------------------------------------------------------------------
# LiveRelativeVolumeCalculator
# ---------------------------------------------------------------------------

class TestLiveRelativeVolumeCalculator:
    def _make_calc(self, daily_bars=None, quote=None, lookback=10):
        cfg = _make_live_config(lookback=lookback)
        data = MagicMock()

        if daily_bars is None:
            daily_bars = _make_daily_bar_df([100_000] * (lookback + 1))
        if quote is None:
            quote = {'volume': 200_000}

        data.get_bars.return_value = daily_bars
        data.get_quote_data.return_value = {'AAPL': quote}
        logger = MagicMock()
        return LiveRelativeVolumeCalculator(cfg, data, logger)

    def test_normal_rvol_calculation(self):
        calc = self._make_calc(
            daily_bars=_make_daily_bar_df([100_000] * 11),
            quote={'volume': 200_000},
        )
        result = calc.calculate_batch(['AAPL'])
        assert 'AAPL' in result
        assert result['AAPL'] == pytest.approx(2.0)

    def test_symbol_skipped_when_no_quote(self):
        cfg = _make_live_config()
        data = MagicMock()
        data.get_bars.return_value = _make_daily_bar_df([100_000] * 11)
        data.get_quote_data.return_value = {}  # no quote
        calc = LiveRelativeVolumeCalculator(cfg, data, MagicMock())
        result = calc.calculate_batch(['AAPL'])
        assert 'AAPL' not in result

    def test_symbol_skipped_when_empty_bars(self):
        cfg = _make_live_config()
        data = MagicMock()
        data.get_bars.return_value = pd.DataFrame()
        data.get_quote_data.return_value = {'AAPL': {'volume': 200_000}}
        calc = LiveRelativeVolumeCalculator(cfg, data, MagicMock())
        result = calc.calculate_batch(['AAPL'])
        assert 'AAPL' not in result

    def test_symbol_skipped_when_fewer_than_5_valid_days(self):
        # 4 non-zero days + rest zero
        vols = [100_000] * 4 + [0] * 7
        calc = self._make_calc(daily_bars=_make_daily_bar_df(vols))
        result = calc.calculate_batch(['AAPL'])
        assert 'AAPL' not in result

    def test_symbol_skipped_when_zero_current_volume(self):
        calc = self._make_calc(quote={'volume': 0})
        result = calc.calculate_batch(['AAPL'])
        assert 'AAPL' not in result

    def test_api_exception_skipped_gracefully(self):
        cfg = _make_live_config()
        data = MagicMock()
        data.get_bars.side_effect = Exception("API error")
        data.get_quote_data.return_value = {'AAPL': {'volume': 200_000}}
        calc = LiveRelativeVolumeCalculator(cfg, data, MagicMock())
        result = calc.calculate_batch(['AAPL'])
        assert 'AAPL' not in result

    def test_multiple_symbols_independent(self):
        cfg = _make_live_config(lookback=10)
        data = MagicMock()
        bars = _make_daily_bar_df([100_000] * 11)
        data.get_bars.return_value = bars
        data.get_quote_data.side_effect = lambda syms: {
            s: {'volume': 200_000 if s == 'AAPL' else 0} for s in syms
        }
        calc = LiveRelativeVolumeCalculator(cfg, data, MagicMock())
        result = calc.calculate_batch(['AAPL', 'MSFT'])
        assert 'AAPL' in result
        assert 'MSFT' not in result  # zero volume

    def test_empty_symbol_list_returns_empty_dict(self):
        calc = self._make_calc()
        result = calc.calculate_batch([])
        assert result == {}


# ---------------------------------------------------------------------------
# BacktestRelativeVolumeCalculator
# ---------------------------------------------------------------------------

class TestBacktestRelativeVolumeCalculator:
    def _make_calc(self, daily_stats=None, intraday_volume=200_000, lookback=10):
        cfg = _make_backtest_config(lookback=lookback)
        data = MagicMock()
        if daily_stats is None:
            daily_stats = {'avg_volume_10d': 100_000}
        data.get_daily_stats.return_value = daily_stats
        intraday_df = pd.DataFrame({'volume': [intraday_volume]})
        data.get_intraday_bars.return_value = intraday_df
        logger = MagicMock()
        return BacktestRelativeVolumeCalculator(cfg, data, logger)

    def test_uses_avg_volume_10d_when_present(self):
        calc = self._make_calc(daily_stats={'avg_volume_10d': 100_000})
        day = datetime(2024, 1, 2)
        result = calc.calculate_batch(['AAPL'], day)
        assert 'AAPL' in result
        assert result['AAPL'] == pytest.approx(2.0)

    def test_falls_back_to_avg_volume_20d(self):
        # No 10d avg, use 20d
        calc = self._make_calc(daily_stats={'avg_volume_20d': 100_000})
        day = datetime(2024, 1, 2)
        result = calc.calculate_batch(['AAPL'], day)
        assert 'AAPL' in result
        assert result['AAPL'] == pytest.approx(2.0)

    def test_symbol_skipped_when_no_daily_stats(self):
        cfg = _make_backtest_config()
        data = MagicMock()
        data.get_daily_stats.return_value = None
        data.get_intraday_bars.return_value = pd.DataFrame({'volume': [200_000]})
        calc = BacktestRelativeVolumeCalculator(cfg, data, MagicMock())
        result = calc.calculate_batch(['AAPL'], datetime(2024, 1, 2))
        assert 'AAPL' not in result

    def test_symbol_skipped_when_zero_avg_volume(self):
        calc = self._make_calc(daily_stats={'avg_volume_10d': 0})
        day = datetime(2024, 1, 2)
        result = calc.calculate_batch(['AAPL'], day)
        assert 'AAPL' not in result

    def test_symbol_skipped_when_zero_current_volume(self):
        calc = self._make_calc(intraday_volume=0)
        day = datetime(2024, 1, 2)
        result = calc.calculate_batch(['AAPL'], day)
        assert 'AAPL' not in result

    def test_exception_in_calculation_skipped_gracefully(self):
        cfg = _make_backtest_config()
        data = MagicMock()
        data.get_daily_stats.side_effect = Exception("DB error")
        calc = BacktestRelativeVolumeCalculator(cfg, data, MagicMock())
        result = calc.calculate_batch(['AAPL'], datetime(2024, 1, 2))
        assert 'AAPL' not in result

    def test_empty_symbol_list_returns_empty_dict(self):
        calc = self._make_calc()
        result = calc.calculate_batch([], datetime(2024, 1, 2))
        assert result == {}


# ---------------------------------------------------------------------------
# DiagnosticCreator
# ---------------------------------------------------------------------------

class TestDiagnosticCreator:
    def _make_config(self):
        cfg = MagicMock()
        cfg.pattern.CONFLUENCE_MIN_PATTERNS = 2
        cfg.pattern.MIN_STEP_UPS = 2
        cfg.pattern.MIN_ADVANCE_RETENTION = 0.5
        cfg.pattern.PARABOLIC_MIN_ANGLE = 45.0
        cfg.pattern.PARABOLIC_MIN_ACCELERATION = 0.001
        cfg.pattern.PARABOLIC_MIN_VOL_MULTIPLIER = 1.5
        cfg.pattern.BREAKOUT_VOL_MULTIPLIER = 1.5
        return cfg

    def test_create_rejection_has_required_fields(self):
        creator = DiagnosticCreator(self._make_config())
        result = creator.create_rejection(
            date=datetime(2024, 1, 2),
            symbol='AAPL',
            gap_percent=15.0,
            relative_volume=2.5,
            reason='test_reason',
        )
        assert result['symbol'] == 'AAPL'
        assert result['gap_percent'] == 15.0
        assert result['reason'] == 'test_reason'
        assert result['phase'] == 'reject'

    def test_create_rejection_accepts_extra_fields(self):
        creator = DiagnosticCreator(self._make_config())
        result = creator.create_rejection(
            date=datetime(2024, 1, 2),
            symbol='AAPL',
            gap_percent=15.0,
            relative_volume=2.5,
            reason='test',
            custom_field='value',
        )
        assert result['custom_field'] == 'value'
