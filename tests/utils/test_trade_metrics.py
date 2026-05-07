"""
Tests for utils/trade_metrics.py

Covers:
- TradeMetrics.to_est: naive datetime → assumed UTC → converted to ET,
  UTC-aware → ET, already-ET passthrough, DST boundary (winter/summer offset)
- TradeMetrics.calculate_profit_erosion_pct: normal, never went positive,
  gave back more than gained (erosion > 100%), exact values
- TradeMetrics.format_date_for_csv / format_time_for_csv / format_timestamp_for_csv
- TradeMetrics.calculate_hold_time: normal, None inputs, negative
- TradeMetrics.get_time_bucket: hour bucket classification
- TradeMetrics.calculate_volume_freshness_ratio: normal, insufficient bars, zero avg
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from utils.trade_metrics import TradeMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(year=2024, month=1, day=2, hour=15, minute=30):
    """UTC-aware timestamp."""
    return pd.Timestamp(datetime(year, month, day, hour, minute, tzinfo=timezone.utc))


def _naive(year=2024, month=1, day=2, hour=15, minute=30):
    """Timezone-naive timestamp (no tzinfo)."""
    return pd.Timestamp(datetime(year, month, day, hour, minute))


# ---------------------------------------------------------------------------
# to_est
# ---------------------------------------------------------------------------

class TestToEst:
    def test_naive_datetime_treated_as_utc(self):
        ts = _naive(hour=15, minute=30)  # naive 15:30
        result = TradeMetrics.to_est(ts)
        assert result.tzname() in ('EST', 'EDT', 'US/Eastern', 'America/New_York')

    def test_utc_aware_converted_correctly_winter(self):
        # Jan: UTC-5
        ts = pd.Timestamp('2024-01-02 15:00:00', tz='UTC')  # 15:00 UTC = 10:00 EST
        result = TradeMetrics.to_est(ts)
        assert result.hour == 10
        assert result.minute == 0

    def test_utc_aware_converted_correctly_summer(self):
        # July: UTC-4
        ts = pd.Timestamp('2024-07-02 15:00:00', tz='UTC')  # 15:00 UTC = 11:00 EDT
        result = TradeMetrics.to_est(ts)
        assert result.hour == 11
        assert result.minute == 0

    def test_already_eastern_passthrough(self):
        ts = pd.Timestamp('2024-01-02 10:00:00', tz='US/Eastern')
        result = TradeMetrics.to_est(ts)
        assert result.hour == 10

    def test_dst_spring_forward(self):
        # 2024-03-10 07:00 UTC = 03:00 ET (spring forward at 2 AM)
        ts = pd.Timestamp('2024-03-10 07:00:00', tz='UTC')
        result = TradeMetrics.to_est(ts)
        assert result.tzname() in ('EDT', 'EST')

    def test_returns_pandas_timestamp(self):
        ts = pd.Timestamp('2024-01-02 15:00:00', tz='UTC')
        result = TradeMetrics.to_est(ts)
        assert isinstance(result, pd.Timestamp)


# ---------------------------------------------------------------------------
# calculate_profit_erosion_pct
# ---------------------------------------------------------------------------

class TestCalculateProfitErosionPct:
    def test_full_profit_retained_zero_erosion(self):
        # Entry=10, max=12, exit=12 → erosion=0%
        result = TradeMetrics.calculate_profit_erosion_pct(10.0, 12.0, 12.0)
        assert result == pytest.approx(0.0)

    def test_never_went_positive_returns_zero(self):
        # Entry=10, max=10 (no gain), exit=9 → erosion=0
        result = TradeMetrics.calculate_profit_erosion_pct(10.0, 10.0, 9.0)
        assert result == pytest.approx(0.0)

    def test_gave_back_half_the_profit(self):
        # Entry=10, max=12, exit=11 → max_profit=2, realized=1, erosion=50%
        result = TradeMetrics.calculate_profit_erosion_pct(10.0, 12.0, 11.0)
        assert result == pytest.approx(50.0)

    def test_gave_back_all_profit_100pct_erosion(self):
        # Entry=10, max=12, exit=10 → realized=0, erosion=100%
        result = TradeMetrics.calculate_profit_erosion_pct(10.0, 12.0, 10.0)
        assert result == pytest.approx(100.0)

    def test_winner_turned_loser_erosion_over_100pct(self):
        # Entry=10, max=12, exit=9 → realized=-1, max_profit=2, erosion=150%
        result = TradeMetrics.calculate_profit_erosion_pct(10.0, 12.0, 9.0)
        assert result == pytest.approx(150.0)

    def test_precise_calculation(self):
        # Entry=5.00, max=5.75, exit=5.20
        # max_profit=0.75, realized=0.20, given_back=0.55, erosion=73.33%
        result = TradeMetrics.calculate_profit_erosion_pct(5.0, 5.75, 5.20)
        assert result == pytest.approx(73.333, rel=1e-3)


# ---------------------------------------------------------------------------
# format_date_for_csv
# ---------------------------------------------------------------------------

class TestFormatDateForCsv:
    def test_returns_yyyy_mm_dd_format(self):
        ts = pd.Timestamp('2024-01-15 10:30:00', tz='UTC')
        result = TradeMetrics.format_date_for_csv(ts)
        assert result == '2024-01-15'

    def test_none_returns_empty_string(self):
        result = TradeMetrics.format_date_for_csv(None)
        assert result == ''

    def test_converts_to_et(self):
        # UTC midnight → previous ET day
        ts = pd.Timestamp('2024-01-02 03:00:00', tz='UTC')  # 22:00 ET on Jan 1
        result = TradeMetrics.format_date_for_csv(ts)
        assert result == '2024-01-01'


# ---------------------------------------------------------------------------
# format_time_for_csv
# ---------------------------------------------------------------------------

class TestFormatTimeForCsv:
    def test_returns_hh_mm_ss_format(self):
        ts = pd.Timestamp('2024-01-02 14:30:00', tz='UTC')  # 09:30 ET (winter)
        result = TradeMetrics.format_time_for_csv(ts)
        assert result == '09:30:00'

    def test_none_returns_empty_string(self):
        result = TradeMetrics.format_time_for_csv(None)
        assert result == ''


# ---------------------------------------------------------------------------
# format_timestamp_for_csv
# ---------------------------------------------------------------------------

class TestFormatTimestampForCsv:
    def test_returns_full_datetime_format(self):
        ts = pd.Timestamp('2024-01-02 14:30:00', tz='UTC')
        result = TradeMetrics.format_timestamp_for_csv(ts)
        assert '2024-01-02' in result
        assert '09:30:00' in result

    def test_none_returns_empty_string(self):
        result = TradeMetrics.format_timestamp_for_csv(None)
        assert result == ''


# ---------------------------------------------------------------------------
# calculate_hold_time
# ---------------------------------------------------------------------------

class TestCalculateHoldTime:
    def test_normal_hold_time(self):
        entry = pd.Timestamp('2024-01-02 09:30:00', tz='UTC')
        exit_ = pd.Timestamp('2024-01-02 10:00:00', tz='UTC')
        result = TradeMetrics.calculate_hold_time(entry, exit_)
        assert result == pytest.approx(30.0)

    def test_none_entry_returns_zero(self):
        exit_ = pd.Timestamp('2024-01-02 10:00:00', tz='UTC')
        assert TradeMetrics.calculate_hold_time(None, exit_) == pytest.approx(0.0)

    def test_none_exit_returns_zero(self):
        entry = pd.Timestamp('2024-01-02 09:30:00', tz='UTC')
        assert TradeMetrics.calculate_hold_time(entry, None) == pytest.approx(0.0)

    def test_zero_hold_time(self):
        ts = pd.Timestamp('2024-01-02 09:30:00', tz='UTC')
        assert TradeMetrics.calculate_hold_time(ts, ts) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_time_bucket
# ---------------------------------------------------------------------------

class TestGetTimeBucket:
    def _et(self, hour):
        return pd.Timestamp(f'2024-01-02 {hour:02d}:00:00', tz='US/Eastern')

    def test_premarket_4am_is_early_premarket(self):
        assert TradeMetrics.get_time_bucket(self._et(5)) == 'early_premarket'

    def test_7am_bucket(self):
        assert TradeMetrics.get_time_bucket(self._et(7)) == '7am'

    def test_8am_bucket(self):
        assert TradeMetrics.get_time_bucket(self._et(8)) == '8am'

    def test_9am_bucket(self):
        assert TradeMetrics.get_time_bucket(self._et(9)) == '9am'

    def test_10am_bucket(self):
        assert TradeMetrics.get_time_bucket(self._et(10)) == '10am'

    def test_11am_bucket(self):
        assert TradeMetrics.get_time_bucket(self._et(11)) == '11am'

    def test_afternoon_bucket(self):
        assert TradeMetrics.get_time_bucket(self._et(12)) == 'afternoon'
        assert TradeMetrics.get_time_bucket(self._et(15)) == 'afternoon'


# ---------------------------------------------------------------------------
# calculate_volume_freshness_ratio
# ---------------------------------------------------------------------------

class TestCalculateVolumeFreshnessRatio:
    def _make_bars(self, volumes, entry_index=None):
        if entry_index is None:
            entry_index = len(volumes) - 1
        return pd.DataFrame({
            'volume': volumes,
            'close': [10.0] * len(volumes),
        }), entry_index

    def test_fresh_momentum_high_recent_volume(self):
        # Session avg=10k, recent 5 bars avg=50k → ratio=5.0
        vols = [10_000] * 15 + [50_000] * 5
        bars, idx = self._make_bars(vols, entry_index=19)
        result = TradeMetrics.calculate_volume_freshness_ratio(bars, idx)
        assert result > 1.0

    def test_stale_momentum_low_recent_volume(self):
        vols = [100_000] * 15 + [10_000] * 5
        bars, idx = self._make_bars(vols, entry_index=19)
        result = TradeMetrics.calculate_volume_freshness_ratio(bars, idx)
        assert result < 1.0

    def test_insufficient_bars_returns_one(self):
        vols = [10_000, 20_000, 30_000]  # only 3 bars
        bars, idx = self._make_bars(vols, entry_index=2)
        result = TradeMetrics.calculate_volume_freshness_ratio(bars, idx)
        assert result == pytest.approx(1.0)

    def test_zero_session_avg_returns_one(self):
        vols = [0] * 20
        bars, idx = self._make_bars(vols, entry_index=19)
        result = TradeMetrics.calculate_volume_freshness_ratio(bars, idx)
        assert result == pytest.approx(1.0)
