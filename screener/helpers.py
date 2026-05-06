# =====================================================
# screener.helpers.py - Screener helper classes
# =====================================================

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd


class LiveRelativeVolumeCalculator:
    """Calculates relative volume for LIVE trading using API data."""
    
    def __init__(self, config, data_handler, logger):
        self.config = config
        self.data = data_handler  # APIDataHandler
        self.logger = logger
    
    def calculate_batch(self, symbols: List[str]) -> Dict[str, float]:
        """
        Calculate relative volume for live trading using API daily bars.
        
        Uses current volume from quote data vs historical daily average.
        """
        rel_vols = {}
        lookback = self.config.screening.RELATIVE_VOLUME_LOOKBACK_DAYS
        
        for symbol in symbols:
            try:
                # NOTE: Get historical daily bars from API (e.g., Alpaca)
                daily_bars = self.data.get_bars(
                    symbol=symbol,
                    timeframe='1Day',
                    limit=lookback + 1  # +1 to include today
                )
                
                if daily_bars is None or daily_bars.empty or len(daily_bars) < 2:
                    self.logger.debug("%s: Insufficient daily bar history", symbol)
                    continue
                
                # NOTE: Get current quote for real-time volume
                quotes = self.data.get_quote_data([symbol]) or {}
                quote = quotes.get(symbol)
                
                if not quote:
                    self.logger.debug("%s: No quote data available", symbol)
                    continue
                
                current_volume = quote.get('volume', 0)
                
                if current_volume <= 0:
                    continue
                
                # Calculate baseline average (exclude today, filter zero-volume days)
                historical_volumes = daily_bars['volume'].iloc[:-1]  # Exclude today
                valid_vols = historical_volumes[historical_volumes > 0]
                
                if len(valid_vols) < 5:
                    self.logger.debug("%s: Insufficient valid volume days (%d<5)", symbol, len(valid_vols))
                    continue
                
                avg_volume = valid_vols.mean()
                
                if avg_volume <= 0:
                    continue
                
                rel_vol = current_volume / avg_volume
                rel_vols[symbol] = rel_vol
                
                self.logger.debug(
                    "%s: RelVol=%.2fx (current=%d, avg=%.0f)",
                    symbol, rel_vol, current_volume, avg_volume
                )
                
            except Exception as e:
                self.logger.debug("Error calculating live RVOL for %s: %s", symbol, e)
                continue
        
        return rel_vols


class BacktestRelativeVolumeCalculator:
    """Calculates relative volume for BACKTEST using pre-computed aggregates."""
    
    def __init__(self, config, data_handler, logger):
        self.config = config
        self.data = data_handler  # LocalDataHandler
        self.logger = logger
    
    def calculate_batch(self, symbols: List[str], day: datetime) -> Dict[str, float]:
        """
        Calculate relative volume for backtesting WITHOUT lookahead bias.
        
        Uses pre-computed aggregates and only volume up to screening time.
        """
        rel_vols = {}
        
        # NOTE: Calculate screening time (warmup period after session start)
        session_cfg = self.config.session
        if session_cfg.PREMARKET_ENABLED:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            session_start_et = pd.Timestamp(
                day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern'
            )
            warmup = session_cfg.PREMARKET_WARMUP_MINUTES
        else:
            session_start_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')
            warmup = session_cfg.REGULAR_WARMUP_MINUTES
        
        screening_time_et = session_start_et + pd.Timedelta(minutes=warmup)
        screening_time_utc = screening_time_et.tz_convert('UTC')
        
        lookback = self.config.screening.RELATIVE_VOLUME_LOOKBACK_DAYS
        use_10d = lookback <= 10
        
        for symbol in symbols:
            try:
                # NOTE: Use pre-computed average from aggregates (instant!)
                daily_stats = self.data.get_daily_stats(symbol, day)
                
                if daily_stats is None:
                    continue
                
                # NOTE: Get pre-computed average volume
                if use_10d and 'avg_volume_10d' in daily_stats:
                    avg_volume = daily_stats['avg_volume_10d']
                elif 'avg_volume_20d' in daily_stats:
                    avg_volume = daily_stats['avg_volume_20d']
                else:
                    # NOTE: Fallback: Compute manually if aggregates don't have it yet
                    self.logger.debug("%s: Aggregate missing avg_volume, using fallback", symbol)
                    end_date = day - timedelta(days=1)
                    start_date = end_date - timedelta(days=lookback + 10)
                    daily_bars = self.data.get_daily_volume_history(symbol, start_date, end_date, bars=lookback)
                    
                    if daily_bars is None or daily_bars.empty:
                        continue
                    
                    # Filter out zero-volume days
                    valid_vols = daily_bars[daily_bars['volume'] > 0]
                    if len(valid_vols) < 5:
                        continue
                    
                    avg_volume = valid_vols['volume'].mean()
                
                if avg_volume <= 0:
                    continue
                
                # NOTE: Get current day's volume ONLY UP TO SCREENING TIME (no lookahead!)
                current_day_bars = self.data.get_intraday_bars(
                    symbol, 
                    start=day,
                    end=screening_time_utc
                )
                current_volume = current_day_bars['volume'].sum() if not current_day_bars.empty else 0
                
                if current_volume <= 0:
                    continue
                
                rel_vol = current_volume / avg_volume
                rel_vols[symbol] = rel_vol
                
                self.logger.debug(
                    "%s: RelVol=%.2fx (current @%s=%d, avg=%.0f)",
                    symbol, rel_vol, screening_time_et.strftime('%H:%M'), current_volume, avg_volume
                )
                
            except Exception as e:
                self.logger.debug("Error calculating backtest RVOL for %s: %s", symbol, e)
                continue
        
        return rel_vols


class DiagnosticCreator:
    """Creates standardized rejection diagnostics."""
    
    def __init__(self, config):
        self.config = config
    
    def create_rejection(
        self,
        date: datetime,
        symbol: str,
        gap_percent: float,
        relative_volume: Optional[float],
        reason: str,
        **extra_fields
    ) -> Dict:
        """Create a simple rejection diagnostic."""
        diagnostic = {
            "date": pd.Timestamp(date).date(),
            "symbol": symbol,
            "gap_percent": gap_percent,
            "relative_volume": relative_volume,
            "phase": "reject",
            "reason": reason
        }
        diagnostic.update(extra_fields)
        return diagnostic
    
    def create_pattern_rejection(
        self,
        date: datetime,
        symbol: str,
        gap_percent: float,
        relative_volume: Optional[float],
        pattern_data: Dict
    ) -> Dict:
        """Create a detailed pattern rejection diagnostic."""
        step_up = pattern_data.get('step_ups', {}) or {}
        parabolic = pattern_data.get('parabolic', {}) or {}
        breakout = pattern_data.get('breakout', {}) or {}
        volume = pattern_data.get('volume', {}) or {}
        sr = pattern_data.get('support_resistance', {}) or {}
        
        failure_reason = "pattern_invalid"
        failure_details = []
        
        min_score_used = pattern_data.get('min_score_threshold', 15.0)
        
        if pattern_data.get('pattern_strength', 0) < min_score_used:
            failure_details.append(f"low_score({pattern_data.get('pattern_strength', 0):.1f}<{min_score_used})")
        
        if pattern_data.get('pattern_count', 0) < self.config.pattern.CONFLUENCE_MIN_PATTERNS:
            failure_details.append(f"insufficient_patterns({pattern_data.get('pattern_count', 0)}<{self.config.pattern.CONFLUENCE_MIN_PATTERNS})")
        
        if not step_up.get('detected'):
            if step_up.get('step_count', 0) < self.config.pattern.MIN_STEP_UPS:
                failure_details.append(f"step_ups({step_up.get('step_count', 0)}<{self.config.pattern.MIN_STEP_UPS})")
            elif step_up.get('retention_rate', 0) < self.config.pattern.MIN_ADVANCE_RETENTION:
                failure_details.append(f"retention({step_up.get('retention_rate', 0):.1f}<{self.config.pattern.MIN_ADVANCE_RETENTION})")
        
        if not parabolic.get('detected'):
            if parabolic.get('angle', 0) < self.config.pattern.PARABOLIC_MIN_ANGLE:
                failure_details.append(f"angle({parabolic.get('angle', 0):.1f}<{self.config.pattern.PARABOLIC_MIN_ANGLE})")
            elif parabolic.get('acceleration', 0) < self.config.pattern.PARABOLIC_MIN_ACCELERATION:
                failure_details.append(f"accel({parabolic.get('acceleration', 0):.4f}<{self.config.pattern.PARABOLIC_MIN_ACCELERATION})")
            elif parabolic.get('volume_multiplier', 0) < self.config.pattern.PARABOLIC_MIN_VOL_MULTIPLIER:
                failure_details.append(f"vol_mult({parabolic.get('volume_multiplier', 0):.1f}<{self.config.pattern.PARABOLIC_MIN_VOL_MULTIPLIER})")
        
        if not breakout.get('detected'):
            if breakout.get('volume_ratio', 0) < self.config.pattern.BREAKOUT_VOL_MULTIPLIER:
                failure_details.append(f"breakout_vol({breakout.get('volume_ratio', 0):.1f}<{self.config.pattern.BREAKOUT_VOL_MULTIPLIER})")
        
        if failure_details:
            failure_reason = "pattern_invalid:" + ",".join(failure_details)
        
        pattern_count = pattern_data.get('pattern_count', 0)
        retention_value = step_up.get('retention_rate', 0)
        accel_value = parabolic.get('acceleration', 0)
        
        return {
            "date": pd.Timestamp(date).date(),
            "symbol": symbol,
            "gap_percent": gap_percent,
            "relative_volume": relative_volume,
            "phase": "reject",
            "reason": failure_reason,
            "pattern_strength": pattern_data.get('pattern_strength'),
            "min_score_threshold": min_score_used,
            "pattern_count": pattern_count,
            "patterns_detected": "|".join(pattern_data.get('patterns_detected', [])),
            "pa_reason": pattern_data.get('reason'),
            "step_up_detected": step_up.get('detected'),
            "step_up_steps": step_up.get('step_count'),
            "step_up_retention": retention_value,
            "parabolic_detected": parabolic.get('detected'),
            "parabolic_angle": parabolic.get('angle'),
            "parabolic_accel": accel_value,
            "parabolic_vol_mult": parabolic.get('volume_multiplier'),
            "breakout_detected": breakout.get('detected'),
            "breakout_level": breakout.get('breakout_level'),
            "breakout_vol_ratio": breakout.get('volume_ratio'),
            "volume_strength": volume.get('strength'),
            "sr_strength": sr.get('strength'),
            "insufficient_patterns_value": pattern_count,
            "insufficient_patterns_passed": pattern_count >= self.config.pattern.CONFLUENCE_MIN_PATTERNS,
            "retention_value": retention_value,
            "retention_passed": retention_value >= self.config.pattern.MIN_ADVANCE_RETENTION,
            "accel_value": accel_value,
            "accel_passed": accel_value >= self.config.pattern.PARABOLIC_MIN_ACCELERATION,
        }