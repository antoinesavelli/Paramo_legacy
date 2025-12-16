# =====================================================
# data_handler.gap_calculator.py - Unified gap calculation
# =====================================================

import pandas as pd
from datetime import datetime
from typing import Dict, Callable, Optional, Set
import logging

logger = logging.getLogger("data_handler.gap_calculator")


class GapCalculator:
    """Unified gap calculator for both live (API) and backtest (aggregates)."""
    
    def __init__(self, get_daily_stats_func: Callable, file_index: Optional[Dict] = None, 
                 universe_symbols: Optional[Set[str]] = None):
        """
        Args:
            get_daily_stats_func: Function to get daily OHLCV aggregates
                - For backtest: Returns dict with 'open', 'close', 'volume' from aggregate files
                - For live: Returns dict with 'open', 'close' from API daily bars
            file_index: Optional file index for backtest mode (checking data availability)
            universe_symbols: Optional set of symbols for live mode (from universe)
        """
        self._get_daily_stats = get_daily_stats_func
        self._file_index = file_index or {}
        self._universe_symbols = universe_symbols or set()
    
    def calculate_gaps(self, day: datetime) -> Dict:
        """
        Calculate gaps using daily OHLCV data (works for both live and backtest).
        
        For each symbol:
            - Get today's open price
            - Get previous day's close price
            - Calculate gap % = ((open - prev_close) / prev_close) * 100
        
        Args:
            day: Trading day to calculate gaps for
            
        Returns:
            Dictionary with:
                - 'gaps': DataFrame with [symbol, open_price, prev_close, last_price, gap_percent, volume]
                - 'prev_syms': Set of symbols with previous day data
                - 'today_syms': Set of symbols with current day data
                - 'prev_empty': Boolean indicating if previous day has no data
                - 'today_empty': Boolean indicating if current day has no data
                - 'prev_day_used': String date of previous trading day
                - 'lookback_days': Number of days back to find previous trading day
        """
        day = pd.Timestamp(day).normalize()
        today_str = day.strftime("%Y-%m-%d")
        
        # Find previous trading day (up to 7 days back for weekends/holidays)
        prev_day, lookback_count = self._find_previous_trading_day(day)
        prev_str = prev_day.strftime("%Y-%m-%d")
        
        if lookback_count >= 7:
            logger.warning(f"No previous trading day found within 7 days for {today_str}")
            return self._empty_result(today_str, prev_str, lookback_count)
        
        # Get symbols available in both days
        common_symbols = self._get_common_symbols(today_str, prev_str)
        
        if not common_symbols:
            logger.debug(f"No common symbols between {today_str} and {prev_str}")
            return self._empty_result(today_str, prev_str, lookback_count)
        
        # Calculate gaps for all common symbols
        gaps = []
        today_syms = set()
        prev_syms = set()
        
        for symbol in sorted(common_symbols):
            try:
                # Get today's stats (open price)
                today_stats = self._get_daily_stats(symbol, day)
                if not today_stats or 'open' not in today_stats:
                    continue
                
                # Get previous day's stats (close price)
                prev_stats = self._get_daily_stats(symbol, prev_day)
                if not prev_stats or 'close' not in prev_stats:
                    continue
                
                today_open = today_stats['open']
                prev_close = prev_stats['close']
                
                # Validate prices
                if pd.isna(today_open) or pd.isna(prev_close) or prev_close <= 0:
                    continue
                
                # Calculate gap percentage
                gap_pct = ((today_open - prev_close) / prev_close) * 100.0
                
                gaps.append({
                    'symbol': symbol,
                    'open_price': float(today_open),
                    'prev_close': float(prev_close),
                    'last_price': float(today_open),  # Use open as current price
                    'gap_percent': float(gap_pct),
                    'volume': int(today_stats.get('volume', 0))  # Optional: include volume
                })
                
                today_syms.add(symbol)
                prev_syms.add(symbol)
                
            except Exception as e:
                logger.debug(f"Error calculating gap for {symbol}: {e}")
                continue
        
        logger.info(
            f"Gap calculation for {today_str}: {len(gaps)} symbols "
            f"(prev_day={prev_str}, lookback={lookback_count} days)"
        )
        
        return {
            'gaps': pd.DataFrame(gaps),
            'prev_syms': prev_syms,
            'today_syms': today_syms,
            'prev_empty': len(prev_syms) == 0,
            'today_empty': len(today_syms) == 0,
            'prev_day_used': prev_str,
            'lookback_days': lookback_count
        }
    
    def _find_previous_trading_day(self, day: pd.Timestamp) -> tuple:
        """
        Find the most recent trading day before the given day.
        
        Returns:
            (previous_day, lookback_count)
        """
        prev_day = day - pd.Timedelta(days=1)
        lookback_count = 0
        max_lookback = 7
        
        while lookback_count < max_lookback:
            prev_str = prev_day.strftime("%Y-%m-%d")
            
            # Check if data exists for this day
            if self._file_index:
                # Backtest mode: check file index
                if prev_str in self._file_index and len(self._file_index[prev_str]) > 0:
                    break
            else:
                # Live mode: assume all recent days have data (API will handle missing days)
                break
            
            lookback_count += 1
            prev_day = prev_day - pd.Timedelta(days=1)
        
        return prev_day, lookback_count
    
    def _get_common_symbols(self, today_str: str, prev_str: str) -> set:
        """Get symbols that exist in both today and previous day."""
        if self._file_index:
            # Backtest mode: use file index
            today_symbols = self._file_index.get(today_str, set())
            prev_symbols = self._file_index.get(prev_str, set())
            return today_symbols & prev_symbols
        else:
            # Live mode: use universe symbols (API will filter out missing data)
            return self._universe_symbols
    
    def _empty_result(self, today_str: str, prev_str: str = 'N/A', lookback_count: int = 0) -> Dict:
        """Return empty result structure."""
        return {
            'gaps': pd.DataFrame(columns=['symbol', 'open_price', 'prev_close', 'last_price', 'gap_percent', 'volume']),
            'prev_syms': set(),
            'today_syms': set(),
            'prev_empty': True,
            'today_empty': True,
            'prev_day_used': prev_str,
            'lookback_days': lookback_count
        }