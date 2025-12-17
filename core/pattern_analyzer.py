# =====================================================
# pattern_analyzer.py - Enhanced Pattern Recognition System
# =====================================================

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional
from config import TradingConfig
from data_handler.api import APIDataHandler
from data_handler.local import LocalDataHandler
from utils.logging import get_logger

def log_pattern_error(logger, msg, exc):
    logger.error(f"{msg}: {exc}")

class PatternAnalyzer:
    """Advanced pattern recognition for parabolic momentum with multiple pattern types"""

    def __init__(self, config: TradingConfig, data_handler: Union[APIDataHandler, LocalDataHandler]):
        self.config = config
        self.data_handler = data_handler
        self.logger = get_logger(__name__, component="pattern_analyzer")
        self.pattern_library = self._build_pattern_library()
        self.pattern_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _build_pattern_library(self) -> Dict:
        pc = self.config.pattern
        return {
            'parabolic_spike': {
                'min_angle': pc.PARABOLIC_MIN_ANGLE,
                'min_acceleration': pc.PARABOLIC_MIN_ACCELERATION,
                'volume_multiplier': pc.PARABOLIC_MIN_VOL_MULTIPLIER,
                'duration_minutes': (15, 120)
            },
            'breakout': {
                'range_periods': pc.BREAKOUT_LOOKBACK,
                'breakout_volume': pc.BREAKOUT_VOL_MULTIPLIER,
                'price_buffer_pct': pc.BREAKOUT_PRICE_BUFFER_PCT,
                'follow_through': 0.02
            }
        }

    def analyze_pattern(self, symbol: str, bars: Optional[pd.DataFrame] = None, 
                   cache_key: Optional[str] = None,
                   is_premarket: bool = False,
                   gap_percent: Optional[float] = None) -> Dict:
        """Comprehensive pattern analysis with dynamic thresholds based on gap size."""
        try:
            # Adjust thresholds for premarket
            if is_premarket:
                volume_mult_adjustment = 0.5
                min_patterns_required = 1
            else:
                volume_mult_adjustment = 1.0
                min_patterns_required = self.config.pattern.CONFLUENCE_MIN_PATTERNS
        
            if cache_key is None and bars is not None and not bars.empty:
                if 'timestamp' in bars.columns:
                    first_ts = bars['timestamp'].iloc[0]
                    last_ts = bars['timestamp'].iloc[-1]
                    cache_key = f"{symbol}_{first_ts}_{last_ts}_{len(bars)}"
                else:
                    cache_key = None
        
            # Check cache
            if cache_key and cache_key in self.pattern_cache:
                self.cache_hits += 1
                return self.pattern_cache[cache_key]
            
            self.cache_misses += 1
            
            if bars is None:
                bars = self.data_handler.get_intraday_bars(symbol, '1Min', 390)
            if bars is None or bars.empty or len(bars) < 30:
                return {'valid': False, 'reason': 'Insufficient data'}

            # Log zero-volume bar statistics
            zero_vol_count = (bars['volume'] == 0).sum()
            if zero_vol_count > 0:
                zero_vol_pct = (zero_vol_count / len(bars)) * 100
                self.logger.debug(
                    f"{symbol}: {zero_vol_count}/{len(bars)} bars ({zero_vol_pct:.1f}%) "
                    f"have zero volume (normal for low-volume periods)"
                )
        
            # Work on a local copy
            bars = bars.copy()

            step_up = self._analyze_step_ups(bars)
            volume = self._analyze_volume_pattern(bars)
            sr = self._analyze_support_resistance(bars)
            parabolic = self._detect_parabolic_spike(bars, volume_mult_factor=volume_mult_adjustment)
            breakout = self._detect_breakout(bars)
            confluence = self._calculate_pattern_confluence(step_up, parabolic, breakout, volume, sr)
            
            # ✅ NEW: Dynamic minimum score based on gap percentage
            min_score = self._get_dynamic_min_score(gap_percent)
            
            is_valid = (confluence['total_score'] >= min_score and 
                       confluence['pattern_count'] >= min_patterns_required)
            
            result = {
                'valid': is_valid,
                'symbol': symbol,
                'pattern_strength': confluence['total_score'],
                'patterns_detected': confluence['patterns_detected'],
                'pattern_count': confluence['pattern_count'],
                'min_score_threshold': min_score,  # ✅ NEW: Include threshold used
                'gap_percent': gap_percent,  # ✅ NEW: Include gap for reference
                'step_ups': step_up,
                'parabolic': parabolic,
                'breakout': breakout,
                'volume': volume,
                'support_resistance': sr,
                'timestamp': datetime.now(),
                'is_premarket_analyzed': is_premarket
            }
            
            # ✅ FIXED: Safe string formatting for gap_percent
            gap_str = f"{gap_percent:.1f}%" if gap_percent is not None else "N/A"
            
            self.logger.info(
                f"Pattern analyzed {symbol}: valid={is_valid} "
                f"strength={result['pattern_strength']:.2f} "
                f"min_threshold={min_score:.1f} (gap={gap_str}) "
                f"patterns={result['patterns_detected']}"
            )
            
            # Cache result
            if cache_key:
                if len(self.pattern_cache) > 1000:
                    for key in list(self.pattern_cache.keys())[:100]:
                        del self.pattern_cache[key]
                self.pattern_cache[cache_key] = result
            
            return result
        except Exception as e:
            log_pattern_error(self.logger, f"Error analyzing pattern for {symbol}", e)
            return {'valid': False, 'reason': str(e)}
    
    def _get_dynamic_min_score(self, gap_percent: Optional[float]) -> float:
        """
        Calculate dynamic minimum score threshold based on gap percentage.
        
        Logic:
        - gap >= 100%: min_score = 5.0  (extreme gap, very low bar)
        - gap >= 50%:  min_score = 10.0 (large gap, moderate bar)
        - gap < 50%:   min_score = 15.0 (normal gap, standard bar)
        """
        if gap_percent is None:
            # Fallback to normal threshold if gap not provided
            return self.config.pattern.CONFLUENCE_NORMAL_GAP_MIN_SCORE
        
        pc = self.config.pattern
        
        if gap_percent >= pc.CONFLUENCE_EXTREME_GAP_THRESHOLD:
            return pc.CONFLUENCE_EXTREME_GAP_MIN_SCORE
        elif gap_percent >= pc.CONFLUENCE_LARGE_GAP_THRESHOLD:
            return pc.CONFLUENCE_LARGE_GAP_MIN_SCORE
        else:
            return pc.CONFLUENCE_NORMAL_GAP_MIN_SCORE

    def analyze_pattern_fast(self, symbol: str, gap_percent: float, volume_ratio: float) -> Dict:
        """Simplified pattern analysis for fast backtesting."""
        
        # Simple scoring based on gap and volume
        score = 0
        
        # Gap contribution (50%)
        if gap_percent >= 200:
            score += 50
        elif gap_percent >= 100:
            score += 30
        elif gap_percent >= 50:
            score += 20
        
        # Volume contribution (50%)
        if volume_ratio >= 10:
            score += 50
        elif volume_ratio >= 5:
            score += 30
        elif volume_ratio >= 2:
            score += 20
        
        return {
            'valid': score >= 70,
            'symbol': symbol,
            'pattern_strength': score,
            'patterns_detected': ['momentum'] if score >= 70 else [],
            'pattern_count': 1 if score >= 70 else 0,
            'timestamp': datetime.now()
        }

    def _analyze_step_ups(self, bars: pd.DataFrame) -> Dict:
        if bars.empty:
            return {'detected': False, 'step_count': 0, 'retention_rate': 0}
        highs, lows = self._find_highs_lows(bars)
        step_count, total_advance, total_retained = self._count_step_ups(bars, highs, lows)
        retention_rate = (total_retained / total_advance * 100) if total_advance > 0 else 0
        detected = (step_count >= self.config.pattern.MIN_STEP_UPS and retention_rate >= self.config.pattern.MIN_ADVANCE_RETENTION)
        return {
            'detected': detected,
            'step_count': step_count,
            'retention_rate': retention_rate,
            'total_advance': total_advance,
            'strength': min(100, (step_count / 5) * 50 + (retention_rate / 100) * 50)
        }

    def _find_highs_lows(self, bars: pd.DataFrame):
        highs, lows = [], []
        for i in range(1, len(bars) - 1):
            if bars.iloc[i]['high'] > bars.iloc[i-1]['high'] and bars.iloc[i]['high'] > bars.iloc[i+1]['high']:
                highs.append(i)
            if bars.iloc[i]['low'] < bars.iloc[i-1]['low'] and bars.iloc[i]['low'] < bars.iloc[i+1]['low']:
                lows.append(i)
        return highs, lows

    def _count_step_ups(self, bars: pd.DataFrame, highs: List[int], lows: List[int]):
        step_count, total_advance, total_retained = 0, 0, 0
        for i in range(1, len(highs)):
            prev_high = bars.iloc[highs[i-1]]['high']
            curr_high = bars.iloc[highs[i]]['high']
            if curr_high > prev_high:
                step_count += 1
                advance = curr_high - prev_high
                intermediate_lows = [l for l in lows if highs[i-1] < l < highs[i]]
                if intermediate_lows:
                    lowest = min(bars.iloc[intermediate_lows]['low'])
                    pullback = prev_high - lowest
                    retained = advance - pullback
                    total_advance += advance
                    total_retained += max(0, retained)
        return step_count, total_advance, total_retained

    def _detect_parabolic_spike(self, bars: pd.DataFrame, volume_mult_factor: float = 1.0) -> Dict:
        if len(bars) < 30:
            return {'detected': False, 'strength': 0}
        try:
            prices = bars['close'].values
            times = np.arange(len(prices))
            coeffs = np.polyfit(times[-30:], prices[-30:], 2)
            acceleration = coeffs[0]
            recent_slope = (prices[-1] - prices[-10]) / 10
            angle = np.degrees(np.arctan(recent_slope / prices[-10])) if prices[-10] != 0 else 0
            recent_volume = bars['volume'].iloc[-10:].mean()
            prev_volume = bars['volume'].iloc[-30:-10].mean()
            volume_multiplier = recent_volume / prev_volume * volume_mult_factor if prev_volume > 0 else 0
            cfg = self.pattern_library['parabolic_spike']
            
            # ✅ NEW: Check both minimum and maximum angle thresholds
            angle_valid = (angle >= cfg['min_angle'] and 
                          angle <= self.config.pattern.PARABOLIC_MAX_ANGLE)
            
            detected = (angle_valid and 
                       acceleration >= cfg['min_acceleration'] and 
                       volume_multiplier >= cfg['volume_multiplier'])
            
            # ✅ IMPROVED: Cap strength score at PARABOLIC_MAX_SCORE
            raw_strength = (angle / 90) * 50 + (volume_multiplier / 10) * 50 if detected else 0
            strength = min(self.config.pattern.PARABOLIC_MAX_SCORE, raw_strength)
            
            # ✅ NEW: Log when rejected due to excessive angle
            if not angle_valid and angle > self.config.pattern.PARABOLIC_MAX_ANGLE:
                self.logger.debug(
                    f"Parabolic pattern rejected: angle {angle:.2f}° exceeds maximum "
                    f"{self.config.pattern.PARABOLIC_MAX_ANGLE:.2f}° (potentially unsustainable)"
                )
            
            return {
                'detected': detected,
                'angle': angle,
                'acceleration': acceleration,
                'volume_multiplier': volume_multiplier,
                'strength': strength,
                'angle_valid': angle_valid  # ✅ NEW: Track validity separately
            }
        except Exception as e:
            self.logger.debug(f"Error detecting parabolic spike: {e}")
            return {'detected': False, 'strength': 0}

    def _detect_breakout(self, bars: pd.DataFrame) -> Dict:
        if len(bars) < 30:
            return {'detected': False, 'strength': 0}
        try:
            lookback = self.pattern_library['breakout']['range_periods']
            price_buf = self.pattern_library['breakout']['price_buffer_pct']
            vol_mult = self.pattern_library['breakout']['breakout_volume']

            recent_high = bars['high'].iloc[-lookback:-1].max()
            recent_low = bars['low'].iloc[-lookback:-1].min()
            range_size = recent_high - recent_low
            current_price = bars['close'].iloc[-1]
            current_volume = bars['volume'].iloc[-1]
            avg_volume = bars['volume'].iloc[-lookback:-1].mean()

            breakout_above = current_price > recent_high * (1 + price_buf)
            volume_surge = current_volume > avg_volume * vol_mult
            detected = breakout_above and volume_surge
            strength = min(100, (current_price / recent_high - 1) * 1000 + (current_volume / avg_volume) * 10) if detected else 0
            return {
                'detected': detected,
                'breakout_level': recent_high,
                'range_size': range_size,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 0,
                'strength': strength
            }
        except Exception as e:
            self.logger.debug(f"Error detecting breakout: {e}")
            return {'detected': False, 'strength': 0}

    def _analyze_volume_pattern(self, bars: pd.DataFrame) -> Dict:
        if bars.empty:
            return {'volume_trend': 'unknown', 'avg_volume': 0, 'strength': 0}

        # Use all bars for volume analysis
        vol_ma_5 = bars['volume'].rolling(5, min_periods=1).mean()
        vol_ma_15 = bars['volume'].rolling(15, min_periods=1).mean()

        recent_avg = vol_ma_5.iloc[-1] if not pd.isna(vol_ma_5.iloc[-1]) else 0
        longer_avg = vol_ma_15.iloc[-1] if not pd.isna(vol_ma_15.iloc[-1]) else 0

        if longer_avg == 0:
            ratio = float('inf') if recent_avg > 0 else 1.0
        else:
            ratio = recent_avg / longer_avg

        if ratio > 1.2:
            volume_trend = 'increasing'
            strength = min(100, (ratio - 1) * 50)
        elif ratio < 0.8:
            volume_trend = 'decreasing'
            strength = 0
        else:
            volume_trend = 'stable'
            strength = 30

        high_volume_correlation = self._calculate_high_volume_correlation(bars)
        strength = max(0, (strength + high_volume_correlation) / 2)  # Never go negative
        
        return {
            'volume_trend': volume_trend,
            'avg_volume': bars['volume'].mean(),
            'recent_volume': bars['volume'].iloc[-5:].mean() if len(bars) >= 5 else 0,
            'high_volume_correlation': high_volume_correlation,
            'strength': strength
        }

    def _calculate_high_volume_correlation(self, bars: pd.DataFrame) -> float:
        try:
            # Compute as a local boolean Series; do not assign back into bars
            new_high = bars['high'] == bars['high'].expanding().max()
            high_bars = bars.loc[new_high]
            other_bars = bars.loc[~new_high]
            if len(high_bars) > 0 and len(other_bars) > 0:
                high_avg_volume = high_bars['volume'].mean()
                other_avg_volume = other_bars['volume'].mean()
                if other_avg_volume > 0:
                    return max(0, min((high_avg_volume / other_avg_volume - 1) * 100, 100))
            return 0
        except Exception:
            return 0

    def _analyze_support_resistance(self, bars: pd.DataFrame) -> Dict:
        if bars.empty:
            return {'support': [], 'resistance': [], 'strength': 0}
        support_levels, resistance_levels = self._find_support_resistance_levels(bars)
        current_price = bars['close'].iloc[-1]
        strength = 0
        if resistance_levels and current_price > max(resistance_levels) * 0.99:
            strength += 50
        if support_levels and current_price > max(support_levels) * 1.02:
            strength += 30
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'current_price': current_price,
            'strength': strength
        }

    def _find_support_resistance_levels(self, bars: pd.DataFrame):
        support_levels, resistance_levels = [], []
        for i in range(20, len(bars), 5):
            window = bars.iloc[max(0, i-20):i]
            local_low = window['low'].min()
            tests = sum((window['low'] <= local_low * 1.01) & (window['close'] > local_low))
            if tests >= 2:
                support_levels.append(local_low)
            local_high = window['high'].max()
            tests = sum(window['high'] >= local_high * 0.99)
            if tests >= 2:
                resistance_levels.append(local_high)
        support_levels = sorted(list(set(support_levels)))[-3:]
        resistance_levels = sorted(list(set(resistance_levels)))[-3:]
        return support_levels, resistance_levels

    def _calculate_pattern_confluence(self, step_up: Dict, parabolic: Dict, breakout: Dict, volume: Dict, sr: Dict) -> Dict:
        patterns_detected = []
        total_score = 0.0
        
        # ✅ Load weights from config instead of hardcoding
        pc = self.config.pattern
        weights = {
            'step_up': pc.CONFLUENCE_WEIGHT_STEP_UP,
            'parabolic': pc.CONFLUENCE_WEIGHT_PARABOLIC,
            'breakout': pc.CONFLUENCE_WEIGHT_BREAKOUT,
            'volume': pc.CONFLUENCE_WEIGHT_VOLUME,
            'support_resistance': pc.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE
        }
        
        if step_up.get('detected', False):
            patterns_detected.append('step_up')
            total_score += step_up.get('strength', 0) * weights['step_up']
        if parabolic.get('detected', False):
            patterns_detected.append('parabolic')
            total_score += parabolic.get('strength', 0) * weights['parabolic']
        if breakout.get('detected', False):
            patterns_detected.append('breakout')
            total_score += breakout.get('strength', 0) * weights['breakout']
        total_score += volume.get('strength', 0) * weights['volume']
        total_score += sr.get('strength', 0) * weights['support_resistance']
        
        # ✅ IMPROVED: Cap total confluence score at CONFLUENCE_MAX_SCORE
        capped_score = min(pc.CONFLUENCE_MAX_SCORE, total_score)
        
        return {
            'patterns_detected': patterns_detected,
            'pattern_count': len(patterns_detected),
            'total_score': capped_score
        }