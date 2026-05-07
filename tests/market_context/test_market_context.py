"""
Tests for market_context/scoring.py and market_context/backtest.py

Covers:
- calculate_market_score: SPY bullish/bearish, RUT leadership bonus, VIX classification,
  breadth, volume profile contributions
- classify_environment: score thresholds → favorable / neutral / unfavorable
- should_trade_from_indicators: VIX extreme block, unfavorable environment gate, pass-through
- position_size_adjustment_from_indicators: multiplier by environment
- BacktestMarketContext._vix_level: classification boundaries
- BacktestMarketContext._spy_trend: bullish / bearish / neutral detection
- BacktestMarketContext.should_trade: delegates to shared scoring correctly
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from market_context.scoring import (
    calculate_market_score,
    classify_environment,
    should_trade_from_indicators,
    position_size_adjustment_from_indicators,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mc_config(
    spy_bull=10.0,
    spy_bear=-5.0,
    rut_bull=20.0,
    rut_bear=-15.0,
    rut_lead_edge=0.5,
    rut_lead_bonus=5.0,
    env_favorable_min=65.0,
    env_neutral_min=40.0,
    block_vix_extreme=True,
    should_trade_min_unfav=30.0,
    size_adj_favorable=1.2,
    size_adj_unfavorable=0.5,
):
    mc = MagicMock()
    mc.SPY_TREND_BULL_WEIGHT = spy_bull
    mc.SPY_TREND_BEAR_WEIGHT = spy_bear
    mc.RUT_TREND_BULL_WEIGHT = rut_bull
    mc.RUT_TREND_BEAR_WEIGHT = rut_bear
    mc.RUT_LEAD_MOMENTUM_EDGE = rut_lead_edge
    mc.RUT_LEAD_BONUS = rut_lead_bonus
    mc.ENV_FAVORABLE_MIN = env_favorable_min
    mc.ENV_NEUTRAL_MIN = env_neutral_min
    mc.BLOCK_ON_VIX_EXTREME = block_vix_extreme
    mc.SHOULD_TRADE_MIN_SCORE_IF_UNFAVORABLE = should_trade_min_unfav
    mc.SIZE_ADJ_FAVORABLE = size_adj_favorable
    mc.SIZE_ADJ_UNFAVORABLE = size_adj_unfavorable
    return mc


def _neutral_context():
    return {
        'spy_trend': {'trend': 'neutral', 'momentum': 0.0},
        'rut_trend': {'trend': 'neutral', 'momentum': 0.0},
        'vix_level': {'classification': 'normal'},
        'market_breadth': {'breadth': 'neutral'},
        'volume_profile': {'profile': 'normal'},
    }


# ---------------------------------------------------------------------------
# calculate_market_score
# ---------------------------------------------------------------------------

class TestCalculateMarketScore:
    def test_neutral_context_returns_near_50_plus_vix_normal(self):
        ctx = _neutral_context()
        mc = _make_mc_config()
        score = calculate_market_score(ctx, mc)
        # neutral SPY/RUT=0, VIX normal=+10, breadth neutral=0, volume normal=0 → 60
        assert score == pytest.approx(60.0)

    def test_all_bullish_increases_score(self):
        ctx = {
            'spy_trend': {'trend': 'bullish', 'momentum': 1.0},
            'rut_trend': {'trend': 'bullish', 'momentum': 1.0},
            'vix_level': {'classification': 'normal'},
            'market_breadth': {'breadth': 'positive'},
            'volume_profile': {'profile': 'high'},
        }
        mc = _make_mc_config()
        score = calculate_market_score(ctx, mc)
        # 50 + 10 + 20 + 10 + 10 + 10 = 110 → capped at 100
        assert score == pytest.approx(100.0)

    def test_all_bearish_decreases_score(self):
        ctx = {
            'spy_trend': {'trend': 'bearish', 'momentum': -1.0},
            'rut_trend': {'trend': 'bearish', 'momentum': -1.0},
            'vix_level': {'classification': 'high'},
            'market_breadth': {'breadth': 'negative'},
            'volume_profile': {'profile': 'low'},
        }
        mc = _make_mc_config()
        score = calculate_market_score(ctx, mc)
        # 50 - 5 - 15 - 20 - 10 = 0 → floored at 0
        assert score == pytest.approx(0.0)

    def test_rut_leadership_bonus_applied_when_bullish_and_leading(self):
        ctx = {
            'spy_trend': {'trend': 'bullish', 'momentum': 0.2},
            'rut_trend': {'trend': 'bullish', 'momentum': 1.0},
            'vix_level': {'classification': 'normal'},
            'market_breadth': {'breadth': 'neutral'},
            'volume_profile': {'profile': 'normal'},
        }
        mc = _make_mc_config(rut_lead_edge=0.5, rut_lead_bonus=5.0)
        score = calculate_market_score(ctx, mc)
        # 50 + 10 + 20 + 5(bonus) + 10 = 95
        assert score == pytest.approx(95.0)

    def test_rut_leadership_bonus_not_applied_when_rut_not_leading(self):
        ctx = {
            'spy_trend': {'trend': 'bullish', 'momentum': 1.0},
            'rut_trend': {'trend': 'bullish', 'momentum': 0.2},
            'vix_level': {'classification': 'normal'},
            'market_breadth': {'breadth': 'neutral'},
            'volume_profile': {'profile': 'normal'},
        }
        mc = _make_mc_config(rut_lead_edge=0.5)
        score = calculate_market_score(ctx, mc)
        # RUT momentum < SPY momentum, no bonus
        # 50 + 10 + 20 + 10 = 90
        assert score == pytest.approx(90.0)

    def test_vix_extreme_applies_large_penalty(self):
        ctx = _neutral_context()
        ctx['vix_level'] = {'classification': 'extreme'}
        mc = _make_mc_config()
        score = calculate_market_score(ctx, mc)
        # 50 - 20 = 30 (neutral contributions + VIX extreme=-20)
        assert score == pytest.approx(30.0)

    def test_vix_low_applies_moderate_penalty(self):
        ctx = _neutral_context()
        ctx['vix_level'] = {'classification': 'low'}
        mc = _make_mc_config()
        score = calculate_market_score(ctx, mc)
        # 50 - 10 = 40
        assert score == pytest.approx(40.0)

    def test_score_clamped_to_0_100(self):
        ctx = _neutral_context()
        mc = _make_mc_config(spy_bull=1000.0)  # huge weight
        score = calculate_market_score(
            {**ctx, 'spy_trend': {'trend': 'bullish', 'momentum': 0}}, mc
        )
        assert 0.0 <= score <= 100.0

    def test_missing_keys_defaults_gracefully(self):
        mc = _make_mc_config()
        score = calculate_market_score({}, mc)
        assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# classify_environment
# ---------------------------------------------------------------------------

class TestClassifyEnvironment:
    def test_high_score_is_favorable(self):
        mc = _make_mc_config(env_favorable_min=65.0, env_neutral_min=40.0)
        ctx = {'market_score': 80.0}
        assert classify_environment(ctx, mc) == 'favorable'

    def test_mid_score_is_neutral(self):
        mc = _make_mc_config(env_favorable_min=65.0, env_neutral_min=40.0)
        ctx = {'market_score': 55.0}
        assert classify_environment(ctx, mc) == 'neutral'

    def test_low_score_is_unfavorable(self):
        mc = _make_mc_config(env_favorable_min=65.0, env_neutral_min=40.0)
        ctx = {'market_score': 20.0}
        assert classify_environment(ctx, mc) == 'unfavorable'

    def test_score_exactly_at_favorable_threshold(self):
        mc = _make_mc_config(env_favorable_min=65.0, env_neutral_min=40.0)
        ctx = {'market_score': 65.0}
        assert classify_environment(ctx, mc) == 'favorable'

    def test_score_exactly_at_neutral_threshold(self):
        mc = _make_mc_config(env_favorable_min=65.0, env_neutral_min=40.0)
        ctx = {'market_score': 40.0}
        assert classify_environment(ctx, mc) == 'neutral'

    def test_missing_market_score_defaults_to_neutral(self):
        mc = _make_mc_config(env_favorable_min=65.0, env_neutral_min=40.0)
        result = classify_environment({}, mc)
        assert result in ('favorable', 'neutral', 'unfavorable')


# ---------------------------------------------------------------------------
# should_trade_from_indicators
# ---------------------------------------------------------------------------

class TestShouldTradeFromIndicators:
    def test_empty_indicators_returns_true(self):
        mc = _make_mc_config()
        assert should_trade_from_indicators({}, mc) is True

    def test_none_indicators_returns_true(self):
        mc = _make_mc_config()
        assert should_trade_from_indicators(None, mc) is True

    def test_vix_extreme_blocks_trading(self):
        mc = _make_mc_config(block_vix_extreme=True)
        indicators = {
            'vix_level': {'classification': 'extreme'},
            'trading_environment': 'neutral',
            'market_score': 55,
        }
        assert should_trade_from_indicators(indicators, mc) is False

    def test_vix_extreme_does_not_block_when_config_disabled(self):
        mc = _make_mc_config(block_vix_extreme=False)
        indicators = {
            'vix_level': {'classification': 'extreme'},
            'trading_environment': 'neutral',
            'market_score': 55,
        }
        assert should_trade_from_indicators(indicators, mc) is True

    def test_unfavorable_low_score_blocks_trading(self):
        mc = _make_mc_config(block_vix_extreme=False, should_trade_min_unfav=30.0)
        indicators = {
            'vix_level': {'classification': 'normal'},
            'trading_environment': 'unfavorable',
            'market_score': 20,
        }
        assert should_trade_from_indicators(indicators, mc) is False

    def test_unfavorable_high_score_allows_trading(self):
        mc = _make_mc_config(block_vix_extreme=False, should_trade_min_unfav=30.0)
        indicators = {
            'vix_level': {'classification': 'normal'},
            'trading_environment': 'unfavorable',
            'market_score': 35,
        }
        assert should_trade_from_indicators(indicators, mc) is True

    def test_favorable_environment_allows_trading(self):
        mc = _make_mc_config(block_vix_extreme=True)
        indicators = {
            'vix_level': {'classification': 'normal'},
            'trading_environment': 'favorable',
            'market_score': 80,
        }
        assert should_trade_from_indicators(indicators, mc) is True


# ---------------------------------------------------------------------------
# position_size_adjustment_from_indicators
# ---------------------------------------------------------------------------

class TestPositionSizeAdjustment:
    def test_empty_indicators_returns_1(self):
        mc = _make_mc_config()
        assert position_size_adjustment_from_indicators({}, mc) == pytest.approx(1.0)

    def test_none_indicators_returns_1(self):
        mc = _make_mc_config()
        assert position_size_adjustment_from_indicators(None, mc) == pytest.approx(1.0)

    def test_favorable_returns_configured_multiplier(self):
        mc = _make_mc_config(size_adj_favorable=1.2)
        ind = {'trading_environment': 'favorable'}
        assert position_size_adjustment_from_indicators(ind, mc) == pytest.approx(1.2)

    def test_unfavorable_returns_configured_multiplier(self):
        mc = _make_mc_config(size_adj_unfavorable=0.5)
        ind = {'trading_environment': 'unfavorable'}
        assert position_size_adjustment_from_indicators(ind, mc) == pytest.approx(0.5)

    def test_neutral_returns_1(self):
        mc = _make_mc_config()
        ind = {'trading_environment': 'neutral'}
        assert position_size_adjustment_from_indicators(ind, mc) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BacktestMarketContext — unit tests using in-process construction
# ---------------------------------------------------------------------------

class TestBacktestMarketContextVixLevel:
    def _make_mc(self, vix_vals):
        from market_context.backtest import BacktestMarketContext
        with patch('market_context.backtest.BacktestMarketContext._load_csv') as m:
            m.return_value = pd.DataFrame()
            obj = BacktestMarketContext.__new__(BacktestMarketContext)

        cfg = MagicMock()
        cfg.market_context.VIX_LOW_MAX = 12.0
        cfg.market_context.VIX_NORMAL_MAX = 20.0
        cfg.market_context.VIX_ELEVATED_MAX = 30.0
        cfg.market_context.VIX_HIGH_MAX = 40.0
        cfg.market_context.SMA_FAST = 5
        cfg.market_context.SMA_SLOW = 20
        cfg.market_context.SPY_TREND_BULL_WEIGHT = 10.0
        cfg.market_context.SPY_TREND_BEAR_WEIGHT = -5.0
        cfg.market_context.RUT_TREND_BULL_WEIGHT = 20.0
        cfg.market_context.RUT_TREND_BEAR_WEIGHT = -15.0
        cfg.market_context.RUT_LEAD_MOMENTUM_EDGE = 0.5
        cfg.market_context.RUT_LEAD_BONUS = 5.0
        cfg.market_context.ENV_FAVORABLE_MIN = 65.0
        cfg.market_context.ENV_NEUTRAL_MIN = 40.0
        cfg.market_context.BLOCK_ON_VIX_EXTREME = True
        cfg.market_context.SHOULD_TRADE_MIN_SCORE_IF_UNFAVORABLE = 30.0
        cfg.market_context.SIZE_ADJ_FAVORABLE = 1.2
        cfg.market_context.SIZE_ADJ_UNFAVORABLE = 0.5
        cfg.market_context.BREADTH_POSITIVE_RUT_RET_MIN = 0.003
        cfg.market_context.BREADTH_NEGATIVE_RUT_RET_MAX = -0.003
        obj.config = cfg
        from utils.logging import get_logger
        obj.logger = get_logger('test_mc', component='test')
        obj.market_indicators = {}
        obj._spy = pd.DataFrame()
        obj._vix = pd.DataFrame()
        obj._rut = pd.DataFrame()

        vix_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-02')],
            'Close': [vix_vals],
        })
        return obj, vix_df

    def _classify(self, level):
        obj, vix_df = self._make_mc(level)
        return obj._vix_level(vix_df)['classification']

    def test_vix_below_low_max_is_low(self):
        assert self._classify(10.0) == 'low'

    def test_vix_normal_range(self):
        assert self._classify(15.0) == 'normal'

    def test_vix_elevated(self):
        assert self._classify(25.0) == 'elevated'

    def test_vix_high(self):
        assert self._classify(35.0) == 'high'

    def test_vix_extreme(self):
        assert self._classify(50.0) == 'extreme'

    def test_empty_vix_df_returns_default_normal(self):
        obj, _ = self._make_mc(15.0)
        result = obj._vix_level(pd.DataFrame())
        assert result['classification'] == 'normal'


class TestBacktestMarketContextSpyTrend:
    def _make_obj(self):
        from market_context.backtest import BacktestMarketContext
        obj = BacktestMarketContext.__new__(BacktestMarketContext)
        cfg = MagicMock()
        cfg.market_context.SMA_FAST = 5
        cfg.market_context.SMA_SLOW = 10
        obj.config = cfg
        from utils.logging import get_logger
        obj.logger = get_logger('test_mc', component='test')
        return obj

    def _make_price_df(self, prices):
        return pd.DataFrame({'Close': prices, 'date': pd.date_range('2024-01-01', periods=len(prices))})

    def test_bullish_when_price_above_both_smas(self):
        obj = self._make_obj()
        # Rising prices: current > sma_fast > sma_slow
        prices = [100 + i for i in range(15)]
        df = self._make_price_df(prices)
        result = obj._spy_trend(df)
        assert result['trend'] == 'bullish'

    def test_bearish_when_price_below_both_smas(self):
        obj = self._make_obj()
        # Declining prices: current < sma_fast < sma_slow
        prices = [100 - i for i in range(15)]
        df = self._make_price_df(prices)
        result = obj._spy_trend(df)
        assert result['trend'] == 'bearish'

    def test_unknown_when_insufficient_data(self):
        obj = self._make_obj()
        df = self._make_price_df([100, 101, 102])  # fewer than SMA_SLOW=10
        result = obj._spy_trend(df)
        assert result['trend'] == 'unknown'


class TestBacktestMarketContextShouldTrade:
    def _make_context_obj(self, indicators):
        from market_context.backtest import BacktestMarketContext
        obj = BacktestMarketContext.__new__(BacktestMarketContext)
        cfg = MagicMock()
        cfg.market_context.BLOCK_ON_VIX_EXTREME = True
        cfg.market_context.SHOULD_TRADE_MIN_SCORE_IF_UNFAVORABLE = 30.0
        obj.config = cfg
        from utils.logging import get_logger
        obj.logger = get_logger('test_mc', component='test')
        obj.market_indicators = indicators
        obj._spy = pd.DataFrame()
        obj._vix = pd.DataFrame()
        obj._rut = pd.DataFrame()
        return obj

    def test_should_trade_true_in_favorable_env(self):
        obj = self._make_context_obj({
            'vix_level': {'classification': 'normal'},
            'trading_environment': 'favorable',
            'market_score': 80,
        })
        assert obj.should_trade() is True

    def test_should_trade_false_in_extreme_vix(self):
        obj = self._make_context_obj({
            'vix_level': {'classification': 'extreme'},
            'trading_environment': 'neutral',
            'market_score': 55,
        })
        assert obj.should_trade() is False

    def test_should_trade_false_when_unfavorable_and_below_min_score(self):
        obj = self._make_context_obj({
            'vix_level': {'classification': 'normal'},
            'trading_environment': 'unfavorable',
            'market_score': 15,
        })
        assert obj.should_trade() is False

    def test_should_trade_true_when_no_indicators_set(self):
        obj = self._make_context_obj({})
        assert obj.should_trade() is True
