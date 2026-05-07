# =====================================================
# market_context.live.py - Market Context Analysis (Live)
# =====================================================

from datetime import datetime
from typing import Dict
import pandas as pd
import numpy as np
from utils.logging import get_logger
from market_context.scoring import (
    calculate_market_score,
    classify_environment,
    should_trade_from_indicators,
    position_size_adjustment_from_indicators,
)

class MarketContext:
    """Analyzes broad market conditions and context (live via API)"""

    def __init__(self, config, api):
        self.config = config
        self.api = api
        self.logger = get_logger(__name__, component="market_context")
        self.market_indicators: Dict = {}
        self.last_update = None

    def update_market_context(self) -> Dict:
        """Update all market context indicators"""
        try:
            context = {
                'spy_trend': self._analyze_spy_trend(),
                'rut_trend': self._analyze_rut_trend(),
                'vix_level': self._get_vix_level(),
                'market_breadth': self._calculate_market_breadth(),
                'sector_rotation': self._analyze_sector_rotation(),
                'volume_profile': self._analyze_volume_profile(),
                'timestamp': datetime.now()
            }
            context['market_score'] = self._calculate_market_score(context)
            context['trading_environment'] = self._classify_environment(context)
            self.market_indicators = context
            self.last_update = datetime.now()
            return context
        except Exception as e:
            self.logger.error("Error updating market context: %s", e)
            return self.market_indicators

    def _analyze_spy_trend(self) -> Dict:
        """Analyze SPY trend and momentum."""
        return self._analyze_index_trend(self.config.market_context.SPY_SYMBOL, "SPY")

    def _analyze_rut_trend(self) -> Dict:
        """Analyze RUT/IWM trend and momentum."""
        return self._analyze_index_trend(self.config.market_context.RUT_SYMBOL, "RUT")

    def _analyze_index_trend(self, symbol: str, label: str) -> Dict:
        """Analyze trend and momentum for any index symbol."""
        try:
            bars = self.api.get_bars(symbol, '1Day', limit=max(20, self.config.market_context.SMA_SLOW)).df
            if bars.empty:
                return {'trend': 'unknown', 'strength': 0}
            bars['sma_fast'] = bars['close'].rolling(self.config.market_context.SMA_FAST).mean()
            bars['sma_slow'] = bars['close'].rolling(self.config.market_context.SMA_SLOW).mean()
            current_price = float(bars['close'].iloc[-1])
            sma_f = float(bars['sma_fast'].iloc[-1])
            sma_s = float(bars['sma_slow'].iloc[-1])
            if current_price > sma_f > sma_s:
                trend, strength = 'bullish', min(100.0, ((current_price / sma_s - 1.0) * 100.0) * 10.0)
            elif current_price < sma_f < sma_s:
                trend, strength = 'bearish', min(100.0, ((1.0 - current_price / sma_s) * 100.0) * 10.0)
            else:
                trend, strength = 'neutral', 50.0
            momentum = float((bars['close'].iloc[-1] / bars['close'].iloc[-self.config.market_context.SMA_FAST] - 1) * 100.0) if len(bars) >= self.config.market_context.SMA_FAST + 1 else 0.0
            return {'trend': trend, 'strength': strength, 'momentum': momentum, 'price': current_price, 'sma_5': sma_f, 'sma_20': sma_s}
        except Exception as e:
            self.logger.error("Error analyzing %s trend: %s", label, e)
            return {'trend': 'unknown', 'strength': 0}

    def _get_vix_level(self) -> Dict:
        """Get VIX level and classification"""
        try:
            bars = self.api.get_bars(self.config.market_context.VIX_SYMBOL, '1Day', limit=1).df
            if bars.empty:
                return {'level': 20.0, 'classification': 'normal'}
            vix_level = float(bars['close'].iloc[-1])
            mc = self.config.market_context
            if vix_level < mc.VIX_LOW_MAX:
                classification = 'low'
            elif vix_level < mc.VIX_NORMAL_MAX:
                classification = 'normal'
            elif vix_level < mc.VIX_ELEVATED_MAX:
                classification = 'elevated'
            elif vix_level < mc.VIX_HIGH_MAX:
                classification = 'high'
            else:
                classification = 'extreme'
            return {'level': vix_level, 'classification': classification}
        except Exception as e:
            self.logger.error("Error getting VIX level: %s", e, exc_info=True)
            return {'level': 20.0, 'classification': 'normal'}

    def _calculate_market_breadth(self) -> Dict:
        """Calculate market breadth indicators (proxy via sample advances/declines)"""
        try:
            assets = self.api.list_assets(status='active')
            sample_symbols = [a.symbol for a in assets[:100] if getattr(a, "tradable", False)]
            advances, declines = 0, 0
            for symbol in sample_symbols[:50]:  # Limit for performance
                try:
                    bars = self.api.get_bars(symbol, '1Day', limit=2).df
                    if len(bars) >= 2:
                        advances += 1 if float(bars['close'].iloc[-1]) > float(bars['close'].iloc[-2]) else 0
                        declines += 1 if float(bars['close'].iloc[-1]) <= float(bars['close'].iloc[-2]) else 0
                except Exception:
                    pass
            total = advances + declines
            ratio = advances / total if total > 0 else 0.5
            return {
                'advances': advances,
                'declines': declines,
                'ratio': ratio,
                'breadth': 'positive' if ratio > 0.6 else 'negative' if ratio < 0.4 else 'neutral'
            }
        except Exception as e:
            self.logger.error("Error calculating market breadth: %s", e, exc_info=True)
            return {'advances': 0, 'declines': 0, 'ratio': 0.5, 'breadth': 'neutral'}

    def _analyze_sector_rotation(self) -> Dict:
        """Analyze sector rotation patterns"""
        sectors = {
            'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare', 'XLE': 'Energy',
            'XLI': 'Industrials', 'XLY': 'Consumer Discretionary', 'XLP': 'Consumer Staples', 'XLU': 'Utilities'
        }
        sector_performance: Dict[str, float] = {}
        try:
            for ticker, name in sectors.items():
                try:
                    bars = self.api.get_bars(ticker, '1Day', limit=5).df
                    if not bars.empty:
                        performance = (float(bars['close'].iloc[-1]) / float(bars['close'].iloc[0]) - 1.0) * 100.0
                        sector_performance[name] = float(performance)
                except Exception:
                    sector_performance[name] = 0.0
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
            return {'leading': sorted_sectors[:2], 'lagging': sorted_sectors[-2:], 'all_sectors': sector_performance}
        except Exception as e:
            self.logger.error("Error analyzing sector rotation: %s", e, exc_info=True)
            return {'leading': [], 'lagging': [], 'all_sectors': {}}

    def _analyze_volume_profile(self) -> Dict:
        """Analyze overall market volume profile"""
        try:
            bars = self.api.get_bars(self.config.market_context.SPY_SYMBOL, '1Day', limit=20).df
            if bars.empty:
                return {'profile': 'normal', 'relative_volume': 1.0}
            current_volume = float(bars['volume'].iloc[-1])
            avg_volume = float(bars['volume'].iloc[:-1].mean())
            rv = current_volume / avg_volume if avg_volume > 0 else 1.0
            profile = 'high' if rv > 1.5 else 'low' if rv < 0.7 else 'normal'
            return {'profile': profile, 'relative_volume': rv, 'current': current_volume, 'average': avg_volume}
        except Exception as e:
            self.logger.error("Error analyzing volume profile: %s", e, exc_info=True)
            return {'profile': 'normal', 'relative_volume': 1.0}

    def _calculate_market_score(self, context: Dict) -> float:
        return calculate_market_score(context, self.config.market_context)

    def _classify_environment(self, context: Dict) -> str:
        return classify_environment(context, self.config.market_context)

    def should_trade(self) -> bool:
        return should_trade_from_indicators(self.market_indicators, self.config.market_context)

    def get_position_size_adjustment(self) -> float:
        return position_size_adjustment_from_indicators(self.market_indicators, self.config.market_context)