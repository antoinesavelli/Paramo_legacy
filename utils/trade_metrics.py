# =====================================================
# trade_metrics.py - Trade Analysis and Calculation Utilities
# =====================================================

from typing import Dict
import pandas as pd
from utils.logging import get_logger


class TradeMetrics:
    """Utility class for trade-related calculations and metrics."""
    
    def __init__(self, config, pattern_analyzer):
        self.config = config
        self.pattern_analyzer = pattern_analyzer
        self.logger = get_logger(__name__, component="trade_metrics")
    
    @staticmethod
    def to_est(ts) -> pd.Timestamp:
        """
        Convert timestamp to US/Eastern timezone.
        
        Args:
            ts: Timestamp to convert (can be naive or timezone-aware)
            
        Returns:
            Timestamp in US/Eastern timezone
        """
        t = pd.Timestamp(ts)
        
        # If naive, assume UTC
        if t.tzinfo is None:
            t = t.tz_localize('UTC')
        
        # Convert to Eastern Time
        return t.tz_convert('US/Eastern')
    
    @staticmethod
    def get_time_bucket(entry_time_et: pd.Timestamp) -> str:
        """
        Classify entry time into hourly buckets for regime analysis.
        
        Args:
            entry_time_et: Entry timestamp in Eastern Time
            
        Returns:
            String bucket name
        """
        hour = entry_time_et.hour
        
        if hour < 7:
            return "early_premarket"  # 4-7 AM (rare, illiquid)
        elif hour == 7:
            return "7am"  # 7-8 AM (YOUR EDGE)
        elif hour == 8:
            return "8am"  # 8-9 AM
        elif hour == 9:
            return "9am"  # 9-10 AM (includes open)
        elif hour == 10:
            return "10am"  # 10-11 AM
        elif hour == 11:
            return "11am"  # 11-12 PM
        else:
            return "afternoon"  # 12+ PM (avoid)
    
    @staticmethod
    def get_gap_type(bars: pd.DataFrame, entry_index: int, entry_time: pd.Timestamp, gap_pct: float) -> str:
        """
        Determine if gap qualified premarket or intraday.
        
        Premarket gap: Qualified before 9:30 AM (overnight gap)
        Intraday gap: Qualified during regular hours (fresh momentum)
        
        Args:
            bars: All intraday bars
            entry_index: Index of entry bar
            entry_time: Entry timestamp
            gap_pct: Gap percentage
            
        Returns:
            "premarket" or "intraday"
        """
        market_open_et = entry_time.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Convert entry time to ET if needed
        entry_et = TradeMetrics.to_est(entry_time)
        
        # Simple logic: If entered before market open, it's a premarket gap
        if entry_et < market_open_et:
            return "premarket"
        else:
            return "intraday"
    
    @staticmethod
    def calculate_volume_freshness_ratio(bars: pd.DataFrame, entry_index: int) -> float:
        """
        Calculate volume freshness ratio.
        
        Ratio of recent 5-bar volume to session average volume.
        > 2.0 = Fresh momentum (recent volume 2x average)
        1.0-2.0 = Moderate freshness
        < 1.0 = Stale momentum
        
        Args:
            bars: All bars from session start to entry
            entry_index: Index of entry bar
            
        Returns:
            Float freshness ratio
        """
        try:
            if entry_index < 4:
                # Not enough bars for 5-bar average
                return 1.0
            
            # Recent volume (last 5 bars including entry)
            recent_bars = bars.iloc[max(0, entry_index-4):entry_index+1]
            recent_volume_avg = recent_bars['volume'].mean()
            
            # Session average (all bars from start to entry)
            session_bars = bars.iloc[:entry_index+1]
            session_volume_avg = session_bars['volume'].mean()
            
            if session_volume_avg <= 0:
                return 1.0
            
            # Freshness ratio
            freshness_ratio = recent_volume_avg / session_volume_avg
            
            return float(freshness_ratio)
            
        except Exception:
            return 1.0
    
    @staticmethod
    def calculate_profit_erosion_pct(entry_price: float, highest_price: float, exit_price: float) -> float:
        """
        Calculate percentage of maximum profit that was given back.
        
        Example:
          Entry: $5.00
          Max: $5.75 (max profit = $0.75, 15%)
          Exit: $5.20 (realized profit = $0.20, 4%)
          
          Erosion = ($0.75 - $0.20) / $0.75 = 73.3%
          (Gave back 73% of max profit)
        
        Args:
            entry_price: Entry price
            highest_price: Maximum price reached
            exit_price: Exit price
            
        Returns:
            Float erosion percentage (can be >100% if winner turned loser)
        """
        max_profit = highest_price - entry_price
        realized_profit = exit_price - entry_price
        
        if max_profit <= 0:
            return 0.0  # Never went positive
        
        profit_given_back = max_profit - realized_profit
        erosion_pct = (profit_given_back / max_profit) * 100.0
        
        return float(erosion_pct)
    
    @staticmethod
    def calculate_hold_time(entry_ts, exit_ts) -> float:
        """Calculate hold time in minutes."""
        try:
            if entry_ts is None or exit_ts is None:
                return 0.0
            entry = pd.Timestamp(entry_ts)
            exit = pd.Timestamp(exit_ts)
            return (exit - entry).total_seconds() / 60.0
        except Exception:
            return 0.0
    
    def extract_pattern_metrics(
        self, 
        symbol: str, 
        bars: pd.DataFrame,
        warmup: int,
        is_premarket: bool
    ) -> Dict:
        """Extract detailed pattern metrics for a symbol."""
        pattern_metrics = {}
        
        if not bars.empty and len(bars) >= warmup:
            bars_warm = bars.iloc[:warmup].copy()
            pa = self.pattern_analyzer.analyze_pattern(
                symbol, bars=bars_warm, is_premarket=is_premarket
            )
            
            # Extract detailed metrics
            step_up = pa.get('step_ups', {}) or {}
            parabolic = pa.get('parabolic', {}) or {}
            breakout = pa.get('breakout', {}) or {}
            volume = pa.get('volume', {}) or {}
            sr = pa.get('support_resistance', {}) or {}
            
            pattern_metrics = {
                'pattern_strength': pa.get('pattern_strength', 0),
                'pattern_count': pa.get('pattern_count', 0),
                'patterns_detected': '|'.join(pa.get('patterns_detected', [])),
                
                # Step-up metrics
                'step_up_detected': step_up.get('detected', False),
                'step_up_count': step_up.get('step_count', 0),
                'step_up_retention': step_up.get('retention_rate', 0),
                'step_up_total_advance': step_up.get('total_advance', 0),
                'step_up_strength': step_up.get('strength', 0),
                
                # Parabolic metrics
                'parabolic_detected': parabolic.get('detected', False),
                'parabolic_angle': parabolic.get('angle', 0),
                'parabolic_acceleration': parabolic.get('acceleration', 0),
                'parabolic_vol_multiplier': parabolic.get('volume_multiplier', 0),
                'parabolic_strength': parabolic.get('strength', 0),
                
                # Breakout metrics
                'breakout_detected': breakout.get('detected', False),
                'breakout_level': breakout.get('breakout_level', 0),
                'breakout_range_size': breakout.get('range_size', 0),
                'breakout_vol_ratio': breakout.get('volume_ratio', 0),
                'breakout_strength': breakout.get('strength', 0),
                
                # Volume metrics
                'volume_trend': volume.get('volume_trend', 'unknown'),
                'volume_avg': volume.get('avg_volume', 0),
                'volume_recent': volume.get('recent_volume', 0),
                'volume_high_correlation': volume.get('high_volume_correlation', 0),
                'volume_strength': volume.get('strength', 0),
                
                # Support/Resistance metrics
                'sr_current_price': sr.get('current_price', 0),
                'sr_support_levels': len(sr.get('support', [])),
                'sr_resistance_levels': len(sr.get('resistance', [])),
                'sr_strength': sr.get('strength', 0),
            }
        
        return pattern_metrics
    
    @staticmethod
    def format_timestamp_for_csv(ts) -> str:
        """
        Format timestamp for CSV export with date and time.
        
        Args:
            ts: Timestamp to format
            
        Returns:
            Formatted string in 'YYYY-MM-DD HH:MM:SS' format (EST)
        """
        if ts is None:
            return ""
        
        # Convert to EST if not already
        est_ts = TradeMetrics.to_est(ts)
        
        # Format as 'YYYY-MM-DD HH:MM:SS'
        return est_ts.strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def format_date_for_csv(ts) -> str:
        """
        Format timestamp date for CSV export.
        
        Args:
            ts: Timestamp to format
            
        Returns:
            Formatted string in 'YYYY-MM-DD' format (EST)
        """
        if ts is None:
            return ""
        
        # Convert to EST if not already
        est_ts = TradeMetrics.to_est(ts)
        
        # Format as 'YYYY-MM-DD'
        return est_ts.strftime('%Y-%m-%d')
    
    @staticmethod
    def format_time_for_csv(ts) -> str:
        """
        Format timestamp time for CSV export.
        
        Args:
            ts: Timestamp to format
            
        Returns:
            Formatted string in 'HH:MM:SS' format (EST)
        """
        if ts is None:
            return ""
        
        # Convert to EST if not already
        est_ts = TradeMetrics.to_est(ts)
        
        # Format as 'HH:MM:SS'
        return est_ts.strftime('%H:%M:%S')
