# =====================================================
# data_handler.gap_calculator.py - Unified gap calculation
# =====================================================
"""
Two-tier gap calculation with adaptive monitoring system.

Architecture:
1. Pre-Filter (once per day): Uses daily aggregates (high/low) to eliminate stocks that cannot meet gap requirements
2. Adaptive Monitor (continuous): Dynamic frequency monitoring based on gap progress toward threshold
3. Pattern Analysis (1min): Triggers when stocks are fully qualified

Performance:
- Vectorized batch operations for millions of rows
- Parallel price fetching with thread pools
- LRU cache management for aggregates
- Automatic memory cleanup between days/months
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Callable, Optional, Set, List
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import gc

from data_handler.aggregates.aggregate_prefilter import AggregatePreFilter
from data_handler.gap.gap_monitor import AdaptiveGapMonitor

logger = logging.getLogger("data_handler.gap_calculator")


class GapCalculator:
    """
    Orchestrates two-tier gap filtering and adaptive monitoring.
    
    Workflow:
    1. initialize_day(): Pre-filter using daily aggregate OHLC data
    2. update_at_timestamp(): Adaptive monitoring at dynamic intervals
    3. Pattern analysis: Automatically triggered for qualified stocks
    4. end_of_day_cleanup(): Memory cleanup and statistics logging
    """
    
    def __init__(self, 
                 aggregate_base_dir: str,
                 get_current_price_func: Callable,
                 min_gap_percent: float,
                 min_price: float,
                 max_price: float,
                 monitoring_config: Dict,
                 batch_size: int = 500,
                 max_workers: int = 4):
        """
        Args:
            aggregate_base_dir: Base directory for daily_aggregates/YYYY/YYYYMM.parquet
            get_current_price_func: Function(symbol, timestamp) -> current_price
            min_gap_percent: Minimum gap % requirement (e.g., 50.0)
            min_price: Minimum price threshold (e.g., 2.0)
            max_price: Maximum price threshold (e.g., 20.0)
            monitoring_config: Dict from config.gap_monitoring
            batch_size: Symbols per batch for vectorized operations
            max_workers: Thread pool size for parallel price fetching
        """
        self._aggregate_dir = Path(aggregate_base_dir)
        self._get_price = get_current_price_func
        self._min_gap_pct = min_gap_percent
        self._min_price = min_price
        self._max_price = max_price
        
        # Extract monitoring configuration
        self._intervals = {
            'NEGATIVE_GAP_INTERVAL_MIN': monitoring_config.get('NEGATIVE_GAP_INTERVAL_MIN', 30),
            'LOW_GAP_INTERVAL_MIN': monitoring_config.get('LOW_GAP_INTERVAL_MIN', 15),
            'MID_GAP_INTERVAL_MIN': monitoring_config.get('MID_GAP_INTERVAL_MIN', 10),
            'HIGH_GAP_INTERVAL_MIN': monitoring_config.get('HIGH_GAP_INTERVAL_MIN', 5),
            'QUALIFIED_INTERVAL_MIN': monitoring_config.get('QUALIFIED_INTERVAL_MIN', 1)
        }
        
        self._thresholds = {
            'LOW_THRESHOLD_PCT': monitoring_config.get('LOW_THRESHOLD_PCT', 0.50),
            'HIGH_THRESHOLD_PCT': monitoring_config.get('HIGH_THRESHOLD_PCT', 0.90),
            'DEQUALIFY_THRESHOLD_PCT': monitoring_config.get('DEQUALIFY_THRESHOLD_PCT', 0.75)
        }
        
        self._cache_months = monitoring_config.get('AGGREGATE_CACHE_MONTHS', 2)
        
        # Initialize pre-filter component
        self._prefilter = AggregatePreFilter(
            aggregate_base_dir=aggregate_base_dir,
            min_gap_percent=min_gap_percent,
            min_price=min_price,
            max_price=max_price,
            cache_months=self._cache_months
        )
        
        # Initialize adaptive monitor component
        self._monitor = AdaptiveGapMonitor(
            get_current_price_func=get_current_price_func,
            min_gap_percent=min_gap_percent,
            min_price=min_price,
            max_price=max_price,
            config_intervals=self._intervals,
            config_thresholds=self._thresholds,
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        # State tracking
        self._current_day: Optional[pd.Timestamp] = None
        self._prev_day: Optional[pd.Timestamp] = None
        self._current_month: Optional[str] = None
        self._is_initialized: bool = False
        
        # Daily statistics
        self._daily_stats = {
            'prefilter_passed': 0,
            'qualified_count': 0,
            'total_candidates': 0,
            'pattern_analysis_triggered': 0
        }
    
    def initialize_day(self, current_day: pd.Timestamp,
                       prev_trading_day: pd.Timestamp) -> Dict:
        """
        Initialize gap calculation for a trading day.
        
        Two-tier filtering:
        1. Load previous day closes from aggregates
        2. Run pre-filter: Check if day's HIGH can reach min_gap with price range
        3. Initialize adaptive monitoring for passed stocks
        
        Args:
            current_day: Current trading day
            prev_trading_day: Previous trading day
            
        Returns:
            Dict with:
                - 'success': bool
                - 'prefilter_passed': int
                - 'total_candidates': int
                - 'passed_symbols': Set[str]
                - 'prev_day_used': str
                - 'prefilter_details': Dict[symbol, gap_data]
        """
        self._current_day = current_day
        self._prev_day = prev_trading_day
        
        day_str = current_day.strftime("%Y-%m-%d")
        prev_str = prev_trading_day.strftime("%Y-%m-%d")
        
        logger.info(f"Initializing gap calculation for {day_str} (prev: {prev_str})")
        
        # Month boundary check - cleanup old aggregate cache
        current_month = current_day.strftime("%Y%m")
        if self._current_month and self._current_month != current_month:
            logger.info(f"Month transition: {self._current_month} → {current_month}")
            self._cleanup_old_month(self._current_month)
        self._current_month = current_month
        
        # Step 1: Load previous day closes
        prev_closes = self._prefilter.get_previous_day_closes(prev_trading_day)
        
        if not prev_closes:
            logger.error(f"No previous day closes available for {prev_str}")
            return {
                'success': False,
                'prefilter_passed': 0,
                'total_candidates': 0,
                'passed_symbols': set(),
                'prev_day_used': prev_str,
                'error': 'No previous day close data'
            }
        
        logger.info(f"Loaded {len(prev_closes)} previous day closes from {prev_str}")
        
        # Step 2: Run pre-filter (aggregate-based)
        passed_symbols, prefilter_details = self._prefilter.prefilter_day(
            current_day, prev_closes
        )
        
        if not passed_symbols:
            logger.warning(f"No stocks passed pre-filter for {day_str}")
            return {
                'success': True,
                'prefilter_passed': 0,
                'total_candidates': len(prev_closes),
                'passed_symbols': set(),
                'prev_day_used': prev_str
            }
        
        # Step 3: Initialize adaptive monitoring
        self._monitor.initialize_monitoring(prefilter_details)
        
        self._is_initialized = True
        self._daily_stats['prefilter_passed'] = len(passed_symbols)
        self._daily_stats['total_candidates'] = len(prev_closes)
        
        logger.info(
            f"✓ Initialized: {len(passed_symbols)}/{len(prev_closes)} passed pre-filter "
            f"(gap≥{self._min_gap_pct}%, price ${self._min_price}-${self._max_price})"
        )
        
        return {
            'success': True,
            'prefilter_passed': len(passed_symbols),
            'total_candidates': len(prev_closes),
            'passed_symbols': passed_symbols,
            'prev_day_used': prev_str,
            'prefilter_details': prefilter_details
        }
    
    def update_at_timestamp(self, current_time: pd.Timestamp) -> Dict:
        """
        Update gap monitoring at specific timestamp.
        
        Call this every minute. The system automatically determines which stocks
        to check based on their monitoring tier:
        - Negative gap: every 30min
        - Low gap (<50%): every 15min
        - Mid gap (50-90%): every 10min
        - High gap (≥90%): every 5min
        - Qualified: every 1min + pattern analysis
        
        Args:
            current_time: Current market timestamp
            
        Returns:
            Dict with:
                - 'qualified_symbols': Set[str] - ready for pattern analysis
                - 'newly_qualified': List[str] - just became qualified
                - 'newly_dequalified': List[str] - lost qualification
                - 'tier_counts': Dict[str, int] - distribution
                - 'should_run_pattern_analysis': bool
                - 'timestamp': pd.Timestamp
        """
        if not self._is_initialized:
            logger.warning("GapCalculator not initialized - call initialize_day() first")
            return self._empty_update_result()
        
        # Update monitoring (checks tier schedules)
        update_result = self._monitor.update_monitoring(current_time)
        
        # Get currently qualified symbols
        qualified = self._monitor.get_qualified_symbols()
        
        # Track statistics
        if update_result['qualified']:
            self._daily_stats['qualified_count'] += len(update_result['qualified'])
        
        # Pattern analysis runs every minute for qualified stocks
        should_analyze = len(qualified) > 0 and current_time.minute % 1 == 0
        
        if should_analyze:
            self._daily_stats['pattern_analysis_triggered'] += 1
        
        return {
            'qualified_symbols': qualified,
            'newly_qualified': update_result['qualified'],
            'newly_dequalified': update_result['dequalified'],
            'tier_counts': self._monitor.get_statistics()['tier_counts'],
            'should_run_pattern_analysis': should_analyze,
            'timestamp': current_time
        }
    
    def get_symbol_gap_data(self, symbol: str) -> Optional[Dict]:
        """
        Get current gap data for a specific symbol.
        
        Returns:
            Dict with: prev_close, current_price, gap_pct, tier, last_check
            or None if not monitored
        """
        return self._monitor.get_symbol_gap_data(symbol)
    
    def get_all_gap_data(self) -> Dict[str, Dict]:
        """
        Get gap data for all monitored symbols.
        
        Returns:
            Dict of {symbol: gap_data}
        """
        return {
            symbol: self._monitor.get_symbol_gap_data(symbol)
            for symbol in self._monitor._monitored_stocks.keys()
        }
    
    def remove_symbol(self, symbol: str):
        """
        Remove symbol from monitoring.
        
        Use after trade execution or when eliminating a candidate.
        
        Args:
            symbol: Symbol to remove
        """
        self._monitor.remove_symbol(symbol)
        logger.debug(f"Removed {symbol} from gap monitoring")
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive daily statistics.
        
        Returns:
            Dict with prefilter counts, qualified counts, tier distribution, etc.
        """
        monitor_stats = self._monitor.get_statistics()
        
        return {
            'day': self._current_day.strftime("%Y-%m-%d") if self._current_day else None,
            'prev_day': self._prev_day.strftime("%Y-%m-%d") if self._prev_day else None,
            'total_candidates': self._daily_stats['total_candidates'],
            'prefilter_passed': self._daily_stats['prefilter_passed'],
            'qualified_count': self._daily_stats['qualified_count'],
            'pattern_analysis_triggered': self._daily_stats['pattern_analysis_triggered'],
            'monitor_stats': monitor_stats,
            'current_qualified': len(self._monitor.get_qualified_symbols())
        }
    
    def end_of_day_cleanup(self):
        """
        End of day cleanup - call after market close.
        
        Actions:
        - Clear monitoring state
        - Keep aggregate cache (needed for next day's prev_close)
        - Log daily statistics
        - Force garbage collection
        """
        if not self._current_day:
            logger.warning("No current day set - skipping cleanup")
            return
        
        day_str = self._current_day.strftime("%Y-%m-%d")
        logger.info(f"End of day cleanup for {day_str}")
        
        # Log final statistics
        stats = self.get_statistics()
        logger.info(
            f"Day Summary [{day_str}]: "
            f"{stats['prefilter_passed']}/{stats['total_candidates']} passed prefilter, "
            f"{stats['qualified_count']} qualified throughout day, "
            f"{stats['pattern_analysis_triggered']} pattern analyses triggered"
        )
        
        # Clear monitoring state
        self._monitor.clear_all()
        
        # Reset state
        self._is_initialized = False
        self._daily_stats = {
            'prefilter_passed': 0,
            'qualified_count': 0,
            'total_candidates': 0,
            'pattern_analysis_triggered': 0
        }
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"✓ Cleanup complete for {day_str}")
    
    def _cleanup_old_month(self, old_month: str):
        """
        Clean up aggregate cache for old month.
        
        Called automatically on month transitions.
        
        Args:
            old_month: Month in format 'YYYYMM'
        """
        self._prefilter.cleanup_old_month(old_month)
        gc.collect()
        logger.info(f"Cleaned up old month: {old_month}")
    
    def _empty_update_result(self) -> Dict:
        """Return empty update result structure."""
        return {
            'qualified_symbols': set(),
            'newly_qualified': [],
            'newly_dequalified': [],
            'tier_counts': {},
            'should_run_pattern_analysis': False,
            'timestamp': None
        }
    
    def clear_all_caches(self):
        """
        Clear ALL caches - call only when completely done (end of backtest).
        
        Warning: This clears aggregate cache. Next initialize_day() will reload.
        """
        self._prefilter.clear_cache()
        self._monitor.clear_all()
        gc.collect()
        logger.info("Cleared all caches and forced GC")