"""
Tests for data_handler/gap/gap_calculator.py

Since GapCalculator depends on file-system aggregate data and a live price
function, we test it using heavy mocking of its two internal components:
_prefilter (AggregatePreFilter) and _monitor (AdaptiveGapMonitor).

Covers:
- initialize_day: success path, no prev-closes, empty passed symbols
- update_at_timestamp: not initialized guard, qualified symbols returned,
  should_run_pattern_analysis always True when qualified (minute % 1 == 0 is always 0)
- get_statistics: correct structure
- end_of_day_cleanup: resets state, calls monitor.clear_all
- remove_symbol: delegates to monitor
- _empty_update_result: correct structure
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

from data_handler.gap.gap_calculator import GapCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitoring_config():
    return {
        'NEGATIVE_GAP_INTERVAL_MIN': 30,
        'LOW_GAP_INTERVAL_MIN': 15,
        'MID_GAP_INTERVAL_MIN': 10,
        'HIGH_GAP_INTERVAL_MIN': 5,
        'QUALIFIED_INTERVAL_MIN': 1,
        'LOW_THRESHOLD_PCT': 0.50,
        'HIGH_THRESHOLD_PCT': 0.90,
        'DEQUALIFY_THRESHOLD_PCT': 0.75,
        'AGGREGATE_CACHE_MONTHS': 2,
    }


def _make_gap_calc(prev_closes=None, passed_symbols=None, prefilter_details=None):
    """Create a GapCalculator with mocked internals."""
    with (
        patch('data_handler.gap.gap_calculator.AggregatePreFilter') as MockPrefilter,
        patch('data_handler.gap.gap_calculator.AdaptiveGapMonitor') as MockMonitor,
    ):
        mock_pf = MagicMock()
        mock_pf.get_previous_day_closes.return_value = prev_closes if prev_closes is not None else {
            'AAPL': 150.0, 'MSFT': 300.0, 'GME': 20.0
        }
        mock_pf.prefilter_day.return_value = (
            passed_symbols if passed_symbols is not None else {'AAPL', 'GME'},
            prefilter_details if prefilter_details is not None else {
                'AAPL': {'prev_close': 150.0, 'gap_potential': 55.0},
                'GME': {'prev_close': 20.0, 'gap_potential': 60.0},
            }
        )
        mock_pf.cleanup_old_month = MagicMock()
        MockPrefilter.return_value = mock_pf

        mock_mon = MagicMock()
        mock_mon.get_qualified_symbols.return_value = set()
        mock_mon.update_monitoring.return_value = {'qualified': [], 'dequalified': []}
        mock_mon.get_statistics.return_value = {'tier_counts': {}}
        mock_mon._monitored_stocks = {}
        MockMonitor.return_value = mock_mon

        calc = GapCalculator(
            aggregate_base_dir=r'd:\fake\aggregates',
            get_current_price_func=lambda s, t: 100.0,
            min_gap_percent=50.0,
            min_price=2.0,
            max_price=20.0,
            monitoring_config=_make_monitoring_config(),
        )
        calc._prefilter = mock_pf
        calc._monitor = mock_mon
        return calc, mock_pf, mock_mon


# ---------------------------------------------------------------------------
# initialize_day
# ---------------------------------------------------------------------------

class TestInitializeDay:
    def test_success_returns_correct_structure(self):
        calc, _, _ = _make_gap_calc()
        current = pd.Timestamp('2024-01-02')
        prev    = pd.Timestamp('2024-01-01')
        result = calc.initialize_day(current, prev)
        assert result['success'] is True
        assert result['prefilter_passed'] == 2
        assert result['total_candidates'] == 3
        assert 'passed_symbols' in result

    def test_no_prev_closes_returns_failure(self):
        calc, mock_pf, _ = _make_gap_calc(prev_closes={})
        current = pd.Timestamp('2024-01-02')
        prev    = pd.Timestamp('2024-01-01')
        result = calc.initialize_day(current, prev)
        assert result['success'] is False
        assert result['prefilter_passed'] == 0

    def test_empty_passed_symbols_returns_success_with_zero_count(self):
        calc, _, _ = _make_gap_calc(passed_symbols=set())
        current = pd.Timestamp('2024-01-02')
        prev    = pd.Timestamp('2024-01-01')
        result = calc.initialize_day(current, prev)
        assert result['success'] is True
        assert result['prefilter_passed'] == 0

    def test_sets_is_initialized_flag(self):
        calc, _, _ = _make_gap_calc()
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        assert calc._is_initialized is True

    def test_monitor_initialize_called(self):
        calc, mock_pf, mock_mon = _make_gap_calc()
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        mock_mon.initialize_monitoring.assert_called_once()


# ---------------------------------------------------------------------------
# update_at_timestamp
# ---------------------------------------------------------------------------

class TestUpdateAtTimestamp:
    def test_returns_empty_when_not_initialized(self):
        calc, _, _ = _make_gap_calc()
        # Don't call initialize_day
        result = calc.update_at_timestamp(pd.Timestamp('2024-01-02 09:35:00'))
        assert result['qualified_symbols'] == set()
        assert result['should_run_pattern_analysis'] is False

    def test_returns_qualified_symbols_after_init(self):
        calc, _, mock_mon = _make_gap_calc()
        mock_mon.get_qualified_symbols.return_value = {'AAPL', 'GME'}
        mock_mon.update_monitoring.return_value = {
            'qualified': ['AAPL'], 'dequalified': []
        }
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        result = calc.update_at_timestamp(pd.Timestamp('2024-01-02 09:35:00'))
        assert 'AAPL' in result['qualified_symbols']

    def test_should_run_pattern_analysis_when_qualified(self):
        calc, _, mock_mon = _make_gap_calc()
        mock_mon.get_qualified_symbols.return_value = {'AAPL'}
        mock_mon.update_monitoring.return_value = {'qualified': [], 'dequalified': []}
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        result = calc.update_at_timestamp(pd.Timestamp('2024-01-02 09:35:00'))
        assert result['should_run_pattern_analysis'] is True

    def test_should_not_run_pattern_analysis_when_no_qualified(self):
        calc, _, mock_mon = _make_gap_calc()
        mock_mon.get_qualified_symbols.return_value = set()
        mock_mon.update_monitoring.return_value = {'qualified': [], 'dequalified': []}
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        result = calc.update_at_timestamp(pd.Timestamp('2024-01-02 09:35:00'))
        assert result['should_run_pattern_analysis'] is False

    def test_result_contains_required_keys(self):
        calc, _, _ = _make_gap_calc()
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        result = calc.update_at_timestamp(pd.Timestamp('2024-01-02 09:35:00'))
        assert 'qualified_symbols' in result
        assert 'newly_qualified' in result
        assert 'newly_dequalified' in result
        assert 'tier_counts' in result
        assert 'should_run_pattern_analysis' in result
        assert 'timestamp' in result


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------

class TestGetStatistics:
    def test_stats_structure_after_init(self):
        calc, _, _ = _make_gap_calc()
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        stats = calc.get_statistics()
        assert 'day' in stats
        assert 'total_candidates' in stats
        assert 'prefilter_passed' in stats
        assert 'qualified_count' in stats
        assert 'pattern_analysis_triggered' in stats

    def test_stats_counts_match_initialization(self):
        calc, _, _ = _make_gap_calc()
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        stats = calc.get_statistics()
        assert stats['total_candidates'] == 3
        assert stats['prefilter_passed'] == 2


# ---------------------------------------------------------------------------
# end_of_day_cleanup
# ---------------------------------------------------------------------------

class TestEndOfDayCleanup:
    def test_cleanup_resets_is_initialized(self):
        calc, _, _ = _make_gap_calc()
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        assert calc._is_initialized is True
        calc.end_of_day_cleanup()
        assert calc._is_initialized is False

    def test_cleanup_resets_daily_stats(self):
        calc, _, _ = _make_gap_calc()
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        calc.end_of_day_cleanup()
        assert calc._daily_stats['prefilter_passed'] == 0
        assert calc._daily_stats['qualified_count'] == 0
        assert calc._daily_stats['total_candidates'] == 0

    def test_cleanup_calls_monitor_clear_all(self):
        calc, _, mock_mon = _make_gap_calc()
        calc.initialize_day(pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01'))
        calc.end_of_day_cleanup()
        mock_mon.clear_all.assert_called_once()

    def test_cleanup_without_init_does_not_crash(self):
        calc, _, _ = _make_gap_calc()
        calc.end_of_day_cleanup()  # should not raise


# ---------------------------------------------------------------------------
# remove_symbol
# ---------------------------------------------------------------------------

class TestRemoveSymbol:
    def test_delegates_to_monitor(self):
        calc, _, mock_mon = _make_gap_calc()
        calc.remove_symbol('AAPL')
        mock_mon.remove_symbol.assert_called_once_with('AAPL')


# ---------------------------------------------------------------------------
# _empty_update_result
# ---------------------------------------------------------------------------

class TestEmptyUpdateResult:
    def test_structure_is_correct(self):
        calc, _, _ = _make_gap_calc()
        result = calc._empty_update_result()
        assert result['qualified_symbols'] == set()
        assert result['newly_qualified'] == []
        assert result['newly_dequalified'] == []
        assert result['should_run_pattern_analysis'] is False
        assert result['timestamp'] is None
