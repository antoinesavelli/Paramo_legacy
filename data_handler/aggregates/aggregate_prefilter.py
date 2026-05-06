# =====================================================
# Aggregate-based pre-filter for gap trading candidates.
# =====================================================

"""
This module implements the first-tier filtering using daily aggregate data (OHLC).
It eliminates stocks that cannot possibly meet gap requirements based on the day's
high price, thus reducing the number of stocks that need continuous monitoring.

Performance optimizations:
- Vectorized pandas operations for batch processing
- Minimal memory footprint by loading only necessary columns
- Efficient caching of monthly aggregate files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger("data_handler.aggregate_prefilter")


class AggregatePreFilter:
    """
    Pre-filters stocks using daily aggregate OHLC data.
    
    This is the FIRST tier of filtering - runs once per day before market open
    using the complete day's aggregate data (high/low/open/close).
    
    Filtering criteria:
    1. Price range check: low <= MAX_PRICE and high >= MIN_PRICE
    2. Gap potential check: high >= (prev_close * (1 + MIN_GAP_PERCENT/100))
    
    If a stock passes this prefilter, it moves to continuous monitoring.
    """
    
    def __init__(self, aggregate_base_dir: str, min_gap_percent: float,
                 min_price: float, max_price: float,
                 cache_months: int = 2):
        """
        Args:
            aggregate_base_dir: Base directory for aggregate files
            min_gap_percent: Minimum gap % requirement (e.g., 50.0)
            min_price: Minimum price threshold
            max_price: Maximum price threshold
            cache_months: Number of months to keep in cache
        """
        self._aggregate_dir = Path(aggregate_base_dir)
        self._min_gap_pct = min_gap_percent
        self._min_price = min_price
        self._max_price = max_price
        self._cache_months = cache_months
        
        # Cache: {year_month: DataFrame}
        self._aggregate_cache: Dict[str, pd.DataFrame] = {}
        self._cache_access_order = []  # Track LRU for cache eviction
        
    def prefilter_day(self, current_day: pd.Timestamp,
                      prev_day_closes: Dict[str, float]) -> Tuple[Set[str], Dict[str, Dict]]:
        """
        Pre-filter stocks for a trading day using aggregate OHLC data.
        
        Args:
            current_day: Trading day to filter
            prev_day_closes: Dict of {symbol: prev_close_price}
            
        Returns:
            Tuple of:
                - Set of symbols that passed prefilter
                - Dict of {symbol: {'open', 'high', 'low', 'close', 'prev_close', 'max_gap_pct'}}
        """
        day_str = current_day.strftime("%Y-%m-%d")
        
        # Load aggregate data for current day
        daily_agg = self._load_aggregate_for_day(current_day)
        
        if daily_agg is None or daily_agg.empty:
            logger.warning(f"No aggregate data for {day_str}")
            return set(), {}
        
        # Filter stocks that have previous day close data
        common_symbols = set(daily_agg.index) & set(prev_day_closes.keys())
        
        if not common_symbols:
            logger.warning(f"No common symbols between current day and previous close data")
            return set(), {}
        
        # Create DataFrame for vectorized operations
        df = daily_agg.loc[list(common_symbols)].copy()
        df['prev_close'] = df.index.map(prev_day_closes)
        
        # Vectorized filtering
        passed_symbols, details = self._apply_prefilter_vectorized(df, day_str)
        
        logger.info(
            f"PreFilter {day_str}: {len(passed_symbols)}/{len(common_symbols)} stocks passed "
            f"(price: {self._min_price}-{self._max_price}, gap: {self._min_gap_pct}%)"
        )
        
        return passed_symbols, details
    
    def _apply_prefilter_vectorized(self, df: pd.DataFrame,
                                    day_str: str) -> Tuple[Set[str], Dict[str, Dict]]:
        """
        Apply prefilter criteria using vectorized pandas operations.
        
        Returns:
            Tuple of (passed_symbols, details_dict)
        """
        # Calculate maximum possible gap % for each stock (using day's high)
        df['max_gap_pct'] = ((df['high'] - df['prev_close']) / df['prev_close']) * 100.0
        
        # Filter criteria (vectorized)
        price_filter = (df['low'] <= self._max_price) & (df['high'] >= self._min_price)
        gap_filter = df['max_gap_pct'] >= self._min_gap_pct
        valid_data = df['prev_close'].notna() & (df['prev_close'] > 0)
        
        # Combined filter
        passed_mask = price_filter & gap_filter & valid_data
        
        # Extract passed symbols
        passed_df = df[passed_mask]
        passed_symbols = set(passed_df.index)
        
        # Build details dictionary
        details = {}
        for symbol in passed_symbols:
            row = passed_df.loc[symbol]
            details[symbol] = {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'prev_close': float(row['prev_close']),
                'max_gap_pct': float(row['max_gap_pct']),
                'volume': int(row.get('volume', 0))
            }
        
        # Log statistics
        if len(df) > 0:
            failed_price = (~price_filter & valid_data).sum()
            failed_gap = (~gap_filter & valid_data & price_filter).sum()
            
            logger.debug(
                f"  Failed price filter: {failed_price} | "
                f"Failed gap filter: {failed_gap} | "
                f"Passed: {len(passed_symbols)}"
            )
        
        return passed_symbols, details
    
    def _load_aggregate_for_day(self, day: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Load daily aggregate data for a specific day.
        
        Returns:
            DataFrame with symbol as index and columns: open, high, low, close, volume
        """
        year_month = day.strftime("%Y%m")
        day_str = day.strftime("%Y-%m-%d")
        
        # Load monthly aggregate (with caching)
        monthly_df = self._get_monthly_aggregate(year_month, day.year)
        
        if monthly_df is None:
            return None
        
        # Extract data for specific day
        if isinstance(monthly_df.index, pd.MultiIndex):
            # Multi-index: (date, symbol)
            if day_str in monthly_df.index.get_level_values('date').unique():
                return monthly_df.loc[day_str]
            else:
                logger.debug(f"Date {day_str} not found in aggregate file")
                return None
        else:
            # Single index (shouldn't happen, but handle it)
            logger.warning(f"Unexpected aggregate structure for {year_month}")
            return None
    
    def _get_monthly_aggregate(self, year_month: str, year: int) -> Optional[pd.DataFrame]:
        """
        Get monthly aggregate data with caching.
        
        Args:
            year_month: Format 'YYYYMM'
            year: Year for directory structure
            
        Returns:
            DataFrame with MultiIndex (date, symbol)
        """
        # Check cache
        if year_month in self._aggregate_cache:
            self._update_cache_access(year_month)
            return self._aggregate_cache[year_month]
        
        # Load from disk
        aggregate_path = self._aggregate_dir / str(year) / f"{year_month}.parquet"
        
        if not aggregate_path.exists():
            logger.warning(f"Aggregate file not found: {aggregate_path}")
            return None
        
        try:
            # Load parquet file
            df = pd.read_parquet(aggregate_path)
            
            # Ensure proper index structure
            if 'date' in df.columns and 'symbol' in df.columns:
                df = df.set_index(['date', 'symbol'])
            elif 'date' not in df.index.names or 'symbol' not in df.index.names:
                logger.error(f"Invalid aggregate structure in {aggregate_path}")
                return None
            
            # Add to cache
            self._add_to_cache(year_month, df)
            
            logger.debug(f"Loaded aggregate: {aggregate_path} ({len(df)} rows)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading aggregate {aggregate_path}: {e}")
            return None
    
    def _add_to_cache(self, year_month: str, df: pd.DataFrame):
        """Add monthly aggregate to cache with LRU eviction."""
        self._aggregate_cache[year_month] = df
        self._cache_access_order.append(year_month)
        
        # Evict oldest if cache exceeds limit
        if len(self._aggregate_cache) > self._cache_months:
            oldest = self._cache_access_order.pop(0)
            if oldest in self._aggregate_cache:
                del self._aggregate_cache[oldest]
                logger.debug(f"Evicted aggregate from cache: {oldest}")
    
    def _update_cache_access(self, year_month: str):
        """Update LRU access order."""
        if year_month in self._cache_access_order:
            self._cache_access_order.remove(year_month)
        self._cache_access_order.append(year_month)
    
    def get_previous_day_closes(self, prev_day: pd.Timestamp,
                                candidate_symbols: Optional[Set[str]] = None) -> Dict[str, float]:
        """
        Get previous day's closing prices for candidate symbols.
        
        Args:
            prev_day: Previous trading day
            candidate_symbols: Optional set to filter symbols (None = all symbols)
            
        Returns:
            Dict of {symbol: close_price}
        """
        daily_agg = self._load_aggregate_for_day(prev_day)
        
        if daily_agg is None or daily_agg.empty:
            logger.warning(f"No aggregate data for previous day: {prev_day.strftime('%Y-%m-%d')}")
            return {}
        
        # Filter by candidate symbols if provided
        if candidate_symbols:
            available_symbols = set(daily_agg.index) & candidate_symbols
            daily_agg = daily_agg.loc[list(available_symbols)]
        
        # Extract close prices
        closes = daily_agg['close'].to_dict()
        
        # Filter out invalid prices
        closes = {sym: price for sym, price in closes.items()
                  if pd.notna(price) and price > 0}
        
        logger.debug(f"Retrieved {len(closes)} previous day closes for {prev_day.strftime('%Y-%m-%d')}")
        
        return closes
    
    def clear_cache(self):
        """Clear the aggregate cache (call when done with a month)."""
        self._aggregate_cache.clear()
        self._cache_access_order.clear()
        logger.info("Cleared aggregate cache")
    
    def cleanup_old_month(self, current_month: str):
        """
        Remove specific month from cache.
        
        Args:
            current_month: Format 'YYYYMM'
        """
        if current_month in self._aggregate_cache:
            del self._aggregate_cache[current_month]
            if current_month in self._cache_access_order:
                self._cache_access_order.remove(current_month)
            logger.info(f"Removed month {current_month} from cache")
