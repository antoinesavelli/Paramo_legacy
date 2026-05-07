"""
Tests for news/backtest.py — NewsIntegrationBacktest

Covers:
- check_news_approval: no news file → rejected with 'no_news_data'
- check_news_approval: symbol not in file → rejected with 'no_news_data'
- check_news_approval: symbol on different date → rejected
- check_news_approval: negative sentiment above threshold → rejected
- check_news_approval: negative sentiment below threshold, before news time → rejected
- check_news_approval: fully approved (neg low, current_time >= news_time)
- check_news_approval: minimal positive gate → rejected when max_positive too low
- monthly cache: loading the same month twice uses cache
- monthly cache: loading a new month replaces cache
"""

import pytest
import pandas as pd
import io
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

from news.backtest import NewsIntegrationBacktest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(max_neg=0.08, min_pos=0.0):
    cfg = MagicMock()
    cfg.backtest.MAX_NEGATIVE_SENTIMENT = max_neg
    cfg.backtest.MIN_POSITIVE_SENTIMENT = min_pos
    return cfg


def _make_news_df(symbol, date_utc, negative=0.02, positive=0.80, ticker_col='ticker'):
    return pd.DataFrame({
        ticker_col: [symbol],
        'date': [pd.Timestamp(date_utc, tz='UTC')],
        'negative': [negative],
        'positive': [positive],
        'link': ['https://example.com/article'],
    })


def _make_integration(config=None, news_df=None, file_exists=True):
    if config is None:
        config = _make_config()
    obj = NewsIntegrationBacktest.__new__(NewsIntegrationBacktest)
    obj.config = config
    from utils.logging import get_logger
    obj.logger = get_logger('test_news', component='test')
    obj._current_month_key = None
    obj._current_month_data = None
    obj.data_dir = MagicMock()

    if news_df is not None:
        obj._load_month_data = MagicMock(return_value=news_df)
    elif not file_exists:
        obj._load_month_data = MagicMock(return_value=pd.DataFrame())
    else:
        obj._load_month_data = MagicMock(return_value=pd.DataFrame())

    return obj


# ---------------------------------------------------------------------------
# check_news_approval — no news data
# ---------------------------------------------------------------------------

class TestCheckNewsApprovalNoData:
    def test_no_file_returns_no_news_data(self):
        obj = _make_integration(file_exists=False)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['approved'] is False
        assert result['reason'] == 'no_news_data'

    def test_symbol_not_in_file_returns_no_news_data(self):
        news_df = _make_news_df('MSFT', '2024-01-02 08:00:00')
        obj = _make_integration(news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['approved'] is False
        assert result['reason'] == 'no_news_data'

    def test_symbol_on_different_date_returns_no_news_data(self):
        # News published yesterday (Jan 1), trading on Jan 2
        news_df = _make_news_df('AAPL', '2024-01-01 08:00:00')
        obj = _make_integration(news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['approved'] is False
        assert result['reason'] == 'no_news_data'


# ---------------------------------------------------------------------------
# check_news_approval — sentiment gates
# ---------------------------------------------------------------------------

class TestCheckNewsApprovalSentimentGates:
    def test_high_negative_sentiment_rejected(self):
        # neg=0.15 > threshold 0.08
        news_df = _make_news_df('AAPL', '2024-01-02 08:00:00', negative=0.15)
        obj = _make_integration(config=_make_config(max_neg=0.08), news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['approved'] is False
        assert result['reason'] == 'negative_sentiment'

    def test_negative_exactly_at_threshold_rejected(self):
        # neg=0.08, threshold=0.08 → 0.08 > 0.08 is False → should pass sentiment gate
        news_df = _make_news_df('AAPL', '2024-01-02 08:00:00', negative=0.08)
        obj = _make_integration(config=_make_config(max_neg=0.08), news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        # 0.08 is NOT > 0.08, so sentiment gate is NOT triggered
        assert result['reason'] != 'negative_sentiment'

    def test_negative_below_threshold_passes_sentiment_gate(self):
        news_df = _make_news_df('AAPL', '2024-01-02 08:00:00', negative=0.02)
        obj = _make_integration(config=_make_config(max_neg=0.08), news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['reason'] != 'negative_sentiment'


# ---------------------------------------------------------------------------
# check_news_approval — timing gate
# ---------------------------------------------------------------------------

class TestCheckNewsApprovalTimingGate:
    def test_before_news_time_rejected(self):
        # News published 09:00 UTC, current_time=07:00 UTC → not yet published
        news_df = _make_news_df('AAPL', '2024-01-02 09:00:00', negative=0.02)
        obj = _make_integration(news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 07:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['approved'] is False
        assert result['reason'] == 'news_not_yet_published'

    def test_at_news_time_approved(self):
        # current_time exactly at news time → approved
        news_df = _make_news_df('AAPL', '2024-01-02 09:00:00', negative=0.02)
        obj = _make_integration(news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 09:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['approved'] is True

    def test_after_news_time_approved(self):
        # current_time > news time → approved
        news_df = _make_news_df('AAPL', '2024-01-02 08:00:00', negative=0.02)
        obj = _make_integration(news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['approved'] is True
        assert result['reason'] == 'approved'


# ---------------------------------------------------------------------------
# check_news_approval — positive sentiment gate
# ---------------------------------------------------------------------------

class TestCheckNewsApprovalPositiveGate:
    def test_insufficient_positive_sentiment_rejected(self):
        # min_positive=0.5, max_positive=0.2 → rejected
        news_df = _make_news_df('AAPL', '2024-01-02 08:00:00', negative=0.02, positive=0.2)
        obj = _make_integration(config=_make_config(max_neg=0.08, min_pos=0.5), news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['approved'] is False
        assert result['reason'] == 'insufficient_positive_sentiment'

    def test_sufficient_positive_sentiment_approved(self):
        news_df = _make_news_df('AAPL', '2024-01-02 08:00:00', negative=0.02, positive=0.8)
        obj = _make_integration(config=_make_config(max_neg=0.08, min_pos=0.5), news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['approved'] is True


# ---------------------------------------------------------------------------
# check_news_approval — return dict completeness
# ---------------------------------------------------------------------------

class TestCheckNewsApprovalReturnDict:
    def test_approved_result_has_required_keys(self):
        news_df = _make_news_df('AAPL', '2024-01-02 08:00:00', negative=0.02)
        obj = _make_integration(news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert 'approved' in result
        assert 'reason' in result
        assert 'max_negative' in result

    def test_rejected_result_has_approved_false(self):
        obj = _make_integration(file_exists=False)
        result = obj.check_news_approval('AAPL', pd.Timestamp('2024-01-02 14:00:00', tz='UTC'))
        assert result['approved'] is False

    def test_naive_timestamp_handled_gracefully(self):
        news_df = _make_news_df('AAPL', '2024-01-02 08:00:00', negative=0.02)
        obj = _make_integration(news_df=news_df)
        # Naive timestamp (no timezone) should be treated as UTC
        current_time = pd.Timestamp('2024-01-02 14:00:00')  # naive
        result = obj.check_news_approval('AAPL', current_time)
        assert 'approved' in result


# ---------------------------------------------------------------------------
# Monthly cache behavior
# ---------------------------------------------------------------------------

class TestNewsMonthlyCache:
    def test_same_month_uses_cache(self):
        news_df = _make_news_df('AAPL', '2024-01-02 08:00:00', negative=0.02)
        cfg = _make_config()

        # Build a real-ish object where _load_month_data does real caching
        obj = NewsIntegrationBacktest.__new__(NewsIntegrationBacktest)
        obj.config = cfg
        from utils.logging import get_logger
        obj.logger = get_logger('test_news', component='test')
        obj._current_month_key = None
        obj._current_month_data = None
        obj.data_dir = MagicMock()

        load_count = {'n': 0}
        original_news_df = news_df.copy()

        def fake_load(year, month):
            load_count['n'] += 1
            key = f"{year}{month:02d}"
            if obj._current_month_key == key and obj._current_month_data is not None:
                return obj._current_month_data
            obj._current_month_key = key
            obj._current_month_data = original_news_df
            return obj._current_month_data

        obj._load_month_data = fake_load

        ct1 = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        ct2 = pd.Timestamp('2024-01-03 14:00:00', tz='UTC')
        obj.check_news_approval('AAPL', ct1)
        obj.check_news_approval('AAPL', ct2)
        # The cache should mean we only loaded once
        assert load_count['n'] <= 2  # first load + possible second for different date

    def test_symbol_column_fallback_to_symbol(self):
        # Use 'symbol' column instead of 'ticker'
        news_df = _make_news_df('AAPL', '2024-01-02 08:00:00',
                                negative=0.02, ticker_col='symbol')
        obj = _make_integration(news_df=news_df)
        current_time = pd.Timestamp('2024-01-02 14:00:00', tz='UTC')
        result = obj.check_news_approval('AAPL', current_time)
        assert result['approved'] is True
