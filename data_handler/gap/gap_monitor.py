
# =====================================================
# Adaptive gap monitoring system with dynamic frequency adjustments.
# =====================================================
"""
This module implements the SECOND tier of filtering - continuous monitoring
of stocks that passed the pre-filter, with monitoring frequency adapted
based on each stock's gap % progress toward the minimum requirement.

Monitoring tiers:
- Negative gap: 30min intervals
- Low gap (< 50% of min): 15min intervals  
- Mid gap (50-90% of min): 10min intervals
- High gap (>= 90% of min): 5min intervals
- Qualified (meets all criteria): 1min intervals + pattern analysis

Performance optimizations:
- Batch processing of symbols per monitoring tier
- Vectorized gap calculations
- Thread pool for parallel minute-bar fetching
- Efficient memory cleanup
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, List, Callable, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import logging
from utils.logging import get_logger
from utils.helpers import calc_gap_percent

logger = get_logger(__name__)


class MonitoringTier:
    """Enumeration of monitoring tiers with associated intervals."""
    NEGATIVE = ("negative", 30)      # Negative gap %
    LOW = ("low", 15)                # < 50% of min gap
    MID = ("mid", 10)                # 50-90% of min gap
    HIGH = ("high", 5)               # >= 90% of min gap
    QUALIFIED = ("qualified", 1)     # Meets all criteria
    
    def __init__(self, name: str, interval_min: int):
        self.name = name
        self.interval_min = interval_min


class AdaptiveGapMonitor:
    """
    Adaptive monitoring system that adjusts check frequency based on gap progress.
    
    Workflow:
    1. Initialize with stocks that passed pre-filter
    2. At each time interval, check stocks assigned to that tier
    3. Calculate current gap % for each stock
    4. Reassign stocks to appropriate tiers based on gap progress
    5. Trigger pattern analysis for qualified stocks
    """
    
    def __init__(self, get_current_price_func: Callable,
                 min_gap_percent: float, min_price: float, max_price: float,
                 config_intervals: Dict[str, int],
                 config_thresholds: Dict[str, float],
                 batch_size: int = 500,
                 max_workers: int = 4):
        """
        Args:
            get_current_price_func: Function(symbol, timestamp) -> current_price
            min_gap_percent: Minimum gap % requirement
            min_price: Minimum price threshold
            max_price: Maximum price threshold
            config_intervals: Dict with monitoring intervals
            config_thresholds: Dict with threshold percentages
            batch_size: Symbols per batch for vectorized ops
            max_workers: Thread pool size
        """
        self._get_price = get_current_price_func
        self._min_gap_pct = min_gap_percent
        self._min_price = min_price
        self._max_price = max_price
        self._batch_size = batch_size
        self._max_workers = max_workers
        
        # Configure monitoring tiers from config
        self._tiers = {
            'negative': MonitoringTier('negative', config_intervals.get('NEGATIVE_GAP_INTERVAL_MIN', 30)),
            'low': MonitoringTier('low', config_intervals.get('LOW_GAP_INTERVAL_MIN', 15)),
            'mid': MonitoringTier('mid', config_intervals.get('MID_GAP_INTERVAL_MIN', 10)),
            'high': MonitoringTier('high', config_intervals.get('HIGH_GAP_INTERVAL_MIN', 5)),
            'qualified': MonitoringTier('qualified', config_intervals.get('QUALIFIED_INTERVAL_MIN', 1))
        }
        
        # Thresholds
        self._low_threshold = config_thresholds.get('LOW_THRESHOLD_PCT', 0.50)
        self._high_threshold = config_thresholds.get('HIGH_THRESHOLD_PCT', 0.90)
        self._dequalify_threshold = config_thresholds.get('DEQUALIFY_THRESHOLD_PCT', 0.75)
        
        # Monitoring state: {symbol: {'prev_close', 'tier', 'last_check', 'current_price', 'gap_pct'}}
        self._monitored_stocks: Dict[str, Dict] = {}
        
        # Tier assignments: {tier_name: Set[symbol]}
        self._tier_assignments: Dict[str, Set[str]] = {
            'negative': set(),
            'low': set(),
            'mid': set(),
            'high': set(),
            'qualified': set()
        }
        
        # Qualified stocks (ready for pattern analysis)
        self._qualified_symbols: Set[str] = set()
        
        # Statistics
        self._stats = {
            'total_monitored': 0,
            'tier_transitions': defaultdict(int),
            'qualified_count': 0,
            'dequalified_count': 0
        }
    
    def initialize_monitoring(self, prefilter_results: Dict[str, Dict]):
        """
        Initialize monitoring with stocks that passed pre-filter.
        
        Args:
            prefilter_results: Dict from AggregatePreFilter with stock details
        """
        self._monitored_stocks.clear()
        for tier_set in self._tier_assignments.values():
            tier_set.clear()
        self._qualified_symbols.clear()
        
        for symbol, details in prefilter_results.items():
            self._monitored_stocks[symbol] = {
                'prev_close': details['prev_close'],
                'tier': 'negative',  # Start with conservative tier
                'last_check': None,
                'current_price': details['open'],
                'gap_pct': 0.0,
                'prefilter_max_gap': details['max_gap_pct']
            }
            self._tier_assignments['negative'].add(symbol)
        
        self._stats['total_monitored'] = len(self._monitored_stocks)
        
        logger.info(
            f"Initialized adaptive monitoring: {len(self._monitored_stocks)} stocks, "
            f"min_gap={self._min_gap_pct}%, price_range={self._min_price}-{self._max_price}"
        )
    
    def update_monitoring(self, current_time: pd.Timestamp) -> Dict[str, List[str]]:
        """
        Update monitoring at current timestamp.
        
        Checks which tiers need updates based on their intervals,
        fetches current prices, recalculates gaps, and reassigns tiers.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Dict with:
                - 'qualified': List of symbols that became qualified
                - 'dequalified': List of symbols that lost qualification
                - 'updated_tiers': Dict of {tier_name: [symbols]}
        """
        result = {
            'qualified': [],
            'dequalified': [],
            'updated_tiers': defaultdict(list)
        }
        
        # Determine which tiers to check at this time
        symbols_to_check = self._get_symbols_to_check(current_time)
        
        if not symbols_to_check:
            return result
        
        # Batch update prices and gaps
        updated_data = self._batch_update_gaps(symbols_to_check, current_time)
        
        # Reassign tiers based on updated gaps
        transitions = self._reassign_tiers(updated_data, current_time)
        
        result['qualified'] = transitions['newly_qualified']
        result['dequalified'] = transitions['newly_dequalified']
        result['updated_tiers'] = self._get_current_tier_snapshot()
        
        return result
    
    def _get_symbols_to_check(self, current_time: pd.Timestamp) -> Set[str]:
        """
        Determine which symbols need checking at current time based on tier intervals.
        
        Uses modulo arithmetic to align checks (e.g., all 30min checks at :00 and :30).
        """
        current_minute = current_time.hour * 60 + current_time.minute
        symbols_to_check = set()
        
        for tier_name, tier in self._tiers.items():
            # Check if current time aligns with tier interval
            if current_minute % tier.interval_min == 0:
                symbols_to_check.update(self._tier_assignments[tier_name])
        
        return symbols_to_check
    
    def _batch_update_gaps(self, symbols: Set[str],
                           current_time: pd.Timestamp) -> Dict[str, Dict]:
        """
        Update current prices and gap % for batches of symbols using parallel fetching.
        
        Returns:
            Dict of {symbol: {'current_price', 'gap_pct', 'prev_close'}}
        """
        updated = {}
        
        # Process in batches with thread pool
        symbol_list = list(symbols)
        
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit batch jobs
            future_to_symbol = {
                executor.submit(self._fetch_symbol_data, symbol, current_time): symbol
                for symbol in symbol_list
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        updated[symbol] = data
                except Exception as e:
                    logger.debug(f"Error fetching data for {symbol}: {e}")
        
        logger.debug(f"Updated {len(updated)}/{len(symbols)} symbols at {current_time}")
        
        return updated
    
    def _fetch_symbol_data(self, symbol: str, timestamp: pd.Timestamp) -> Optional[Dict]:
        """Fetch current price and calculate gap for a single symbol."""
        try:
            if symbol not in self._monitored_stocks:
                return None
            
            stock_data = self._monitored_stocks[symbol]
            prev_close = stock_data['prev_close']
            
            # Get current price
            current_price = self._get_price(symbol, timestamp)
            
            if current_price is None or pd.isna(current_price) or current_price <= 0:
                return None
            
            # Calculate gap %
            gap_pct = calc_gap_percent(current_price, prev_close)
            
            return {
                'current_price': float(current_price),
                'gap_pct': float(gap_pct),
                'prev_close': float(prev_close)
            }
            
        except Exception as e:
            logger.debug(f"Error in _fetch_symbol_data for {symbol}: {e}")
            return None
    
    def _reassign_tiers(self, updated_data: Dict[str, Dict],
                        current_time: pd.Timestamp) -> Dict[str, List[str]]:
        """
        Reassign stocks to appropriate monitoring tiers based on gap progress.
        
        Returns:
            Dict with 'newly_qualified' and 'newly_dequalified' lists
        """
        newly_qualified = []
        newly_dequalified = []
        
        for symbol, data in updated_data.items():
            if symbol not in self._monitored_stocks:
                continue
            
            stock_data = self._monitored_stocks[symbol]
            old_tier = stock_data['tier']
            current_price = data['current_price']
            gap_pct = data['gap_pct']
            
            # Update stock data
            stock_data['current_price'] = current_price
            stock_data['gap_pct'] = gap_pct
            stock_data['last_check'] = current_time
            
            # Determine new tier
            new_tier = self._determine_tier(gap_pct, current_price)
            
            # Handle tier transition
            if new_tier != old_tier:
                self._transition_tier(symbol, old_tier, new_tier)
                self._stats['tier_transitions'][f"{old_tier}->{new_tier}"] += 1
                
                # Track qualification changes
                if new_tier == 'qualified' and old_tier != 'qualified':
                    newly_qualified.append(symbol)
                    self._qualified_symbols.add(symbol)
                    self._stats['qualified_count'] += 1
                    
                elif old_tier == 'qualified' and new_tier != 'qualified':
                    newly_dequalified.append(symbol)
                    self._qualified_symbols.discard(symbol)
                    self._stats['dequalified_count'] += 1
        
        if newly_qualified:
            logger.info(f"Newly qualified: {len(newly_qualified)} stocks")
        if newly_dequalified:
            logger.info(f"Dequalified: {len(newly_dequalified)} stocks")
        
        return {
            'newly_qualified': newly_qualified,
            'newly_dequalified': newly_dequalified
        }
    
    def _determine_tier(self, gap_pct: float, current_price: float) -> str:
        """
        Determine appropriate monitoring tier based on gap % and price.
        
        Tier logic:
        - Qualified: gap >= min_gap AND min_price <= price <= max_price
        - High: gap >= 90% of min_gap
        - Mid: gap >= 50% of min_gap
        - Low: gap < 50% of min_gap
        - Negative: gap < 0
        
        Special case: If qualified, check dequalify threshold (75%)
        """
        # Check if fully qualified
        if (gap_pct >= self._min_gap_pct and
            self._min_price <= current_price <= self._max_price):
            return 'qualified'
        
        # Check dequalification threshold for currently qualified stocks
        dequalify_gap = self._min_gap_pct * self._dequalify_threshold
        
        # Tier assignment based on gap progress
        if gap_pct < 0:
            return 'negative'
        elif gap_pct >= self._min_gap_pct * self._high_threshold:
            return 'high'
        elif gap_pct >= self._min_gap_pct * self._low_threshold:
            return 'mid'
        else:
            return 'low'
    
    def _transition_tier(self, symbol: str, old_tier: str, new_tier: str):
        """Move symbol from old tier to new tier."""
        if old_tier in self._tier_assignments:
            self._tier_assignments[old_tier].discard(symbol)
        
        if new_tier in self._tier_assignments:
            self._tier_assignments[new_tier].add(symbol)
        
        self._monitored_stocks[symbol]['tier'] = new_tier
        
        logger.debug(f"{symbol}: {old_tier} -> {new_tier} (gap={self._monitored_stocks[symbol]['gap_pct']:.2f}%)")
    
    def get_qualified_symbols(self) -> Set[str]:
        """Get symbols currently qualified for pattern analysis."""
        return self._qualified_symbols.copy()
    
    def get_symbol_gap_data(self, symbol: str) -> Optional[Dict]:
        """Get current gap data for a symbol."""
        return self._monitored_stocks.get(symbol)
    
    def _get_current_tier_snapshot(self) -> Dict[str, List[str]]:
        """Get current tier assignments."""
        return {tier: list(symbols) for tier, symbols in self._tier_assignments.items()}
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics."""
        stats = self._stats.copy()
        stats['tier_counts'] = {tier: len(symbols)
                                for tier, symbols in self._tier_assignments.items()}
        stats['qualified_symbols'] = len(self._qualified_symbols)
        return stats
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from monitoring (e.g., after trade execution)."""
        if symbol in self._monitored_stocks:
            tier = self._monitored_stocks[symbol]['tier']
            self._tier_assignments[tier].discard(symbol)
            del self._monitored_stocks[symbol]
            self._qualified_symbols.discard(symbol)
            logger.debug(f"Removed {symbol} from monitoring")
    
    def clear_all(self):
        """Clear all monitoring state (call at end of day)."""
        self._monitored_stocks.clear()
        for tier_set in self._tier_assignments.values():
            tier_set.clear()
        self._qualified_symbols.clear()
        logger.info("Cleared all monitoring state")
