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
from collections import OrderedDict

# Returned in place of a real detector result when the detector's confluence
# weight is 0.0 — avoids running expensive computation that contributes nothing.
_DETECTOR_DISABLED: Dict = {'detected': False, 'strength': 0, 'disabled': True}

class _LRUCache:
    """Fixed-capacity LRU cache backed by OrderedDict."""

    def __init__(self, maxsize: int = 512):
        self._maxsize = maxsize
        self._store: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key, value) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)  # evict oldest

    def __contains__(self, key):
        return key in self._store

    def __len__(self):
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()

class PatternAnalyzer:
    """Advanced pattern recognition for parabolic momentum with multiple pattern types"""

    def __init__(self, config: TradingConfig, data_handler: Union[APIDataHandler, LocalDataHandler]):
        self.config = config
        self.data_handler = data_handler
        self.logger = get_logger(__name__, component="pattern_analyzer")
        self.pattern_library = self._build_pattern_library()
        self.pattern_cache = _LRUCache(maxsize=512)
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
            if cache_key:
                cached = self.pattern_cache.get(cache_key)
                if cached is not None:
                    self.cache_hits += 1
                    return cached

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
                    "%s: %d/%d bars (%.1f%%) have zero volume (normal for low-volume periods)",
                    symbol, zero_vol_count, len(bars), zero_vol_pct
                )

            # Work on a local copy
            bars = bars.copy()

            pc = self.config.pattern

            step_up  = self._analyze_step_ups(bars)
            volume   = self._analyze_volume_pattern(bars)
            sr       = self._analyze_support_resistance(bars)

            # Only run detectors whose confluence weight is non-zero
            parabolic = (
                self._detect_parabolic_spike(bars, volume_mult_factor=volume_mult_adjustment)
                if pc.CONFLUENCE_WEIGHT_PARABOLIC > 0
                else _DETECTOR_DISABLED
            )
            breakout = (
                self._detect_breakout(bars)
                if pc.CONFLUENCE_WEIGHT_BREAKOUT > 0
                else _DETECTOR_DISABLED
            )

            confluence = self._calculate_pattern_confluence(step_up, parabolic, breakout, volume, sr)

            # Dynamic minimum score based on gap percentage
            min_score = self._get_dynamic_min_score(gap_percent)

            is_valid = (confluence['total_score'] >= min_score and
                       confluence['pattern_count'] >= min_patterns_required)

            # ── Structured failure reason (for diagnostic sub-reason breakdown) ──
            if not is_valid:
                if confluence['total_score'] < min_score:
                    failure_reason = (
                        f"score_below_threshold:"
                        f"score={confluence['total_score']:.1f},"
                        f"threshold={min_score:.1f}"
                    )
                elif confluence['pattern_count'] < min_patterns_required:
                    failure_reason = (
                        f"insufficient_pattern_count:"
                        f"count={confluence['pattern_count']},"
                        f"required={min_patterns_required}"
                    )
                else:
                    failure_reason = 'pattern_invalid'

                su = step_up
                step_sub = []
                if su.get('step_count', 0) < pc.MIN_STEP_UPS:
                    step_sub.append(
                        f"step_count={su.get('step_count', 0)}<{pc.MIN_STEP_UPS}"
                    )
                if su.get('retention_rate', 0) < pc.MIN_ADVANCE_RETENTION:
                    step_sub.append(
                        f"retention={su.get('retention_rate', 0):.1f}%<{pc.MIN_ADVANCE_RETENTION}%"
                    )
                if step_sub:
                    failure_reason += '|step_up:' + ','.join(step_sub)
            else:
                failure_reason = None

            # ── First-qualifying-bar detection (must run BEFORE result dict) ──
            first_signal_bar_idx = -1
            first_signal_ts      = None
            entry_lag_bars       = None

            if is_valid:
                first_signal_bar_idx = self._find_first_qualifying_bar(bars)
                last_bar_idx         = len(bars) - 1

                if first_signal_bar_idx >= 0 and 'timestamp' in bars.columns:
                    raw_ts = bars.iloc[first_signal_bar_idx]['timestamp']
                    first_signal_ts = pd.Timestamp(raw_ts)
                    if first_signal_ts.tz is None:
                        first_signal_ts = first_signal_ts.tz_localize('UTC')
                    first_signal_ts = first_signal_ts.tz_convert('US/Eastern')

                    last_bar_ts_raw = bars.iloc[last_bar_idx]['timestamp']
                    last_bar_ts = pd.Timestamp(last_bar_ts_raw)
                    if last_bar_ts.tz is None:
                        last_bar_ts = last_bar_ts.tz_localize('UTC')
                    last_bar_ts = last_bar_ts.tz_convert('US/Eastern')

                    entry_lag_bars = last_bar_idx - first_signal_bar_idx

                    if entry_lag_bars > 0:
                        self.logger.warning(
                            "[ENTRY LAG] %s: pattern first qualified at bar %d (%s) "
                            "but warmup window ends at bar %d (%s) — %d bar lag (%d min stale)",
                            symbol, first_signal_bar_idx,
                            first_signal_ts.strftime('%H:%M ET'),
                            last_bar_idx, last_bar_ts.strftime('%H:%M ET'),
                            entry_lag_bars, entry_lag_bars
                        )
                    else:
                        self.logger.debug(
                            "[ENTRY LAG] %s: pattern qualified at final warmup bar — no lag",
                            symbol
                        )

            result = {
                'valid':                  is_valid,
                'reason':                 failure_reason,
                'symbol':                 symbol,
                'pattern_strength':       confluence['total_score'],
                'patterns_detected':      confluence['patterns_detected'],
                'pattern_count':          confluence['pattern_count'],
                'min_score_threshold':    min_score,
                'gap_percent':            gap_percent,
                'step_ups':               step_up,
                'parabolic':              parabolic,
                'breakout':               breakout,
                'volume':                 volume,
                'support_resistance':     sr,
                'timestamp':              datetime.now(),
                'is_premarket_analyzed':  is_premarket,
                # ── Entry lag audit fields ──
                'first_signal_bar_idx':   first_signal_bar_idx,
                'first_signal_ts_et':     first_signal_ts.strftime('%H:%M ET') if first_signal_ts else None,
                'entry_lag_bars':         entry_lag_bars,
            }

            gap_str = f"{gap_percent:.1f}%" if gap_percent is not None else "N/A"
            self.logger.info(
                "Pattern analyzed %s: valid=%s strength=%.2f min_threshold=%.1f (gap=%s) patterns=%s",
                symbol, is_valid, result['pattern_strength'], min_score,
                gap_str, result['patterns_detected']
            )

            # Cache result
            if cache_key:
                self.pattern_cache.put(cache_key, result)

            return result
        except Exception as e:
            self.logger.error("Error analyzing pattern for %s: %s", symbol, e)
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

            angle_valid = (angle >= cfg['min_angle'] and
                          angle <= self.config.pattern.PARABOLIC_MAX_ANGLE)

            detected = (angle_valid and
                       acceleration >= cfg['min_acceleration'] and
                       volume_multiplier >= cfg['volume_multiplier'])

            raw_strength = (angle / 90) * 50 + (volume_multiplier / 10) * 50 if detected else 0
            strength = min(self.config.pattern.PARABOLIC_MAX_SCORE, raw_strength)

            if not angle_valid and angle > self.config.pattern.PARABOLIC_MAX_ANGLE:
                self.logger.debug(
                    "Parabolic pattern rejected: angle %.2f° exceeds maximum %.2f° (potentially unsustainable)",
                    angle, self.config.pattern.PARABOLIC_MAX_ANGLE
                )

            return {
                'detected': detected,
                'angle': angle,
                'acceleration': acceleration,
                'volume_multiplier': volume_multiplier,
                'strength': strength,
                'angle_valid': angle_valid
            }
        except Exception as e:
            self.logger.debug("Error detecting parabolic spike: %s", e)
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
            self.logger.debug("Error detecting breakout: %s", e)
            return {'detected': False, 'strength': 0}

    def _analyze_volume_pattern(self, bars: pd.DataFrame) -> Dict:
        if bars.empty:
            return {'volume_trend': 'unknown', 'avg_volume': 0, 'strength': 0}

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
        strength = max(0, (strength + high_volume_correlation) / 2)

        return {
            'volume_trend': volume_trend,
            'avg_volume': bars['volume'].mean(),
            'recent_volume': bars['volume'].iloc[-5:].mean() if len(bars) >= 5 else 0,
            'high_volume_correlation': high_volume_correlation,
            'strength': strength
        }

    def _calculate_high_volume_correlation(self, bars: pd.DataFrame) -> float:
        try:
            new_high = bars['high'] == bars['high'].expanding().max()
            high_bars = bars.loc[new_high]
            other_bars = bars.loc[~new_high]
            if len(high_bars) > 0 and len(other_bars) > 0:
                high_avg_volume = high_bars['volume'].mean()
                other_avg_volume = other_bars['volume'].mean()
                if other_avg_volume > 0:
                    return max(0, min((high_avg_volume / other_avg_volume - 1) * 100, 100))
            return 0
        except Exception as e:
            self.logger.debug("_calculate_high_volume_correlation failed (bars=%d): %s", len(bars), e)
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

    def _calculate_pattern_confluence(self, step_up: Dict, parabolic: Dict, breakout: Dict, volume: Dict, sr: Dict, gap_percent: Optional[float] = None) -> Dict:
        patterns_detected = []
        total_score = 0.0

        pc = self.config.pattern
        weights = {
            'step_up':            pc.CONFLUENCE_WEIGHT_STEP_UP,
            'parabolic':          pc.CONFLUENCE_WEIGHT_PARABOLIC,
            'breakout':           pc.CONFLUENCE_WEIGHT_BREAKOUT,
            'volume':             pc.CONFLUENCE_WEIGHT_VOLUME,
            'support_resistance': pc.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE
        }

        if step_up.get('detected', False) and weights['step_up'] > 0:
            patterns_detected.append('step_up')
            total_score += step_up.get('strength', 0) * weights['step_up']
        # parabolic and breakout are only reached here if their detectors ran
        # (weight > 0). _DETECTOR_DISABLED always has detected=False, so the
        # guard is strictly for correctness when a custom weight is set > 0.
        if parabolic.get('detected', False) and weights['parabolic'] > 0:
            patterns_detected.append('parabolic')
            total_score += parabolic.get('strength', 0) * weights['parabolic']
        if breakout.get('detected', False) and weights['breakout'] > 0:
            patterns_detected.append('breakout')
            total_score += breakout.get('strength', 0) * weights['breakout']
        total_score += volume.get('strength', 0) * weights['volume']
        total_score += sr.get('strength', 0) * weights['support_resistance']

        # Normalize score to 0-100 based on active (non-zero) weights only
        active_weight_sum = sum(w for w in weights.values() if w > 0)
        normalized_score = (total_score / active_weight_sum) if active_weight_sum > 0 else 0.0
        normalized_score = max(0.0, min(100.0, normalized_score))

        # Apply extreme gap penalty if enabled
        penalty_applied = 1.0
        if pc.EXTREME_GAP_PENALTY_ENABLED and gap_percent is not None:
            if gap_percent >= pc.EXTREME_GAP_3000_THRESHOLD:
                penalty_applied = pc.EXTREME_GAP_3000_PENALTY
            elif gap_percent >= pc.EXTREME_GAP_2500_THRESHOLD:
                penalty_applied = pc.EXTREME_GAP_2500_PENALTY
            elif gap_percent >= pc.EXTREME_GAP_2000_THRESHOLD:
                penalty_applied = pc.EXTREME_GAP_2000_PENALTY
            if penalty_applied < 1.0:
                self.logger.debug(
                    "Gap penalty applied: %.1f%% gap → %.1f%% multiplier "
                    "(score reduced from %.2f to %.2f)",
                    gap_percent, penalty_applied * 100,
                    normalized_score, normalized_score * penalty_applied
                )

        final_score = normalized_score * penalty_applied

        return {
            'patterns_detected':   patterns_detected,
            'pattern_count':       len(patterns_detected),
            'total_score':         final_score,
            'score_before_penalty': normalized_score,
            'gap_penalty_applied': penalty_applied
        }

    def _find_first_qualifying_bar(self, bars: pd.DataFrame) -> int:
        """
        Returns the index of the bar that CONFIRMS the qualifying pattern
        (i.e. the bar after the final pivot high is established).
        Entry fill should be at this index + 1 (next bar's open).
        """
        pc = self.config.pattern
        min_steps     = pc.MIN_STEP_UPS
        min_retention = pc.MIN_ADVANCE_RETENTION

        for end_idx in range(3, len(bars) + 1):
            window = bars.iloc[:end_idx]
            highs, lows = self._find_highs_lows(window)
            if len(highs) < 2:
                continue
            step_count, total_advance, total_retained = self._count_step_ups(window, highs, lows)
            retention = (total_retained / total_advance * 100) if total_advance > 0 else 0
            if step_count >= min_steps and retention >= min_retention:
                return end_idx - 1
        return -1