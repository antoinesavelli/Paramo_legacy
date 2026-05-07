
# =====================================================
# Efficient price fetching module for gap monitoring.
# =====================================================

"""
Provides optimized methods to fetch current prices from intraday data files
with batching and caching for performance.
"""

import logging
from utils.logging import get_logger
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Set, Callable
from datetime import datetime

logger = get_logger(__name__)


class IntradayPriceFetcher:
    """
    Fetches current prices from intraday minute-bar data.
    
    Optimizations:
    - Caches loaded daily files
    - Batch lookups using vectorized operations
    - Memory-efficient with automatic cleanup
    """
    
    def __init__(self, data_dir: str, cache_size: int = 2):
        """
        Args:
            data_dir: Base directory for intraday data
            cache_size: Number of daily files to keep in cache
        """
        self._data_dir = Path(data_dir)
        self._cache_size = cache_size
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._cache_order = []
    
    def get_price_at_time(self, symbol: str, timestamp: pd.Timestamp) -> Optional[float]:
        """
        Get price for symbol at specific timestamp.
        
        Args:
            symbol: Stock symbol
            timestamp: Timestamp to fetch price for
            
        Returns:
            Price (close of minute bar) or None if not available
        """
        day_str = timestamp.strftime("%Y-%m-%d")
        
        # Load day data if not cached
        if day_str not in self._daily_cache:
            success = self._load_day_data(day_str)
            if not success:
                return None
        
        day_data = self._daily_cache[day_str]
        
        # Filter for symbol and timestamp
        try:
            if isinstance(day_data.index, pd.MultiIndex):
                # MultiIndex: (timestamp, symbol)
                if timestamp in day_data.index.get_level_values(0):
                    if (timestamp, symbol) in day_data.index:
                        return float(day_data.loc[(timestamp, symbol), 'close'])
            else:
                # Single index or different structure
                mask = (day_data['symbol'] == symbol) & (day_data.index == timestamp)
                if mask.any():
                    return float(day_data.loc[mask, 'close'].iloc[0])
            
            return None
            
        except Exception as e:
            logger.debug("Error fetching price for %s at %s: %s", symbol, timestamp, e)
            return None
    
    def get_prices_batch(self, symbols: Set[str], timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        Get prices for multiple symbols at once (vectorized).
        
        Args:
            symbols: Set of symbols
            timestamp: Timestamp to fetch prices for
            
        Returns:
            Dict of {symbol: price}
        """
        day_str = timestamp.strftime("%Y-%m-%d")
        
        if day_str not in self._daily_cache:
            success = self._load_day_data(day_str)
            if not success:
                return {}
        
        day_data = self._daily_cache[day_str]
        prices = {}
        
        try:
            if isinstance(day_data.index, pd.MultiIndex):
                # MultiIndex: (timestamp, symbol)
                if timestamp in day_data.index.get_level_values(0):
                    time_slice = day_data.loc[timestamp]
                    for symbol in symbols:
                        if symbol in time_slice.index:
                            prices[symbol] = float(time_slice.loc[symbol, 'close'])
            else:
                # Filter by timestamp
                time_slice = day_data[day_data.index == timestamp]
                if not time_slice.empty:
                    for symbol in symbols:
                        symbol_data = time_slice[time_slice['symbol'] == symbol]
                        if not symbol_data.empty:
                            prices[symbol] = float(symbol_data['close'].iloc[0])
            
        except Exception as e:
            logger.debug("Error in batch price fetch at %s: %s", timestamp, e)
        
        return prices
    
    def _load_day_data(self, day_str: str) -> bool:
        """
        Load intraday data for a specific day.
        
        Args:
            day_str: Day in format 'YYYY-MM-DD'
            
        Returns:
            True if loaded successfully
        """
        # Build file path (adjust based on your data structure)
        day_date = pd.Timestamp(day_str)
        year = day_date.strftime("%Y")
        file_path = self._data_dir / year / f"{day_str}.parquet"
        
        if not file_path.exists():
            logger.warning("Intraday data file not found: %s", file_path)
            return False
        
        try:
            df = pd.read_parquet(file_path)
            
            # Ensure proper structure (timestamp, symbol) multi-index
            if 'timestamp' in df.columns and 'symbol' in df.columns:
                df = df.set_index(['timestamp', 'symbol'])
            elif 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            # Add to cache
            self._add_to_cache(day_str, df)
            
            logger.debug("Loaded intraday data: %s (%d rows)", file_path, len(df))
            return True
            
        except Exception as e:
            logger.error("Error loading intraday data %s: %s", file_path, e, exc_info=True)
            return False
    
    def _add_to_cache(self, day_str: str, df: pd.DataFrame):
        """Add day data to cache with LRU eviction."""
        self._daily_cache[day_str] = df
        self._cache_order.append(day_str)
        
        # Evict oldest if exceeds cache size
        if len(self._daily_cache) > self._cache_size:
            oldest = self._cache_order.pop(0)
            if oldest in self._daily_cache:
                del self._daily_cache[oldest]
                logger.debug("Evicted day data from cache: %s", oldest)
    
    def clear_cache(self):
        """Clear all cached data."""
        self._daily_cache.clear()
        self._cache_order.clear()
        logger.info("Cleared intraday price cache")
    
    def cleanup_day(self, day_str: str):
        """Remove specific day from cache."""
        if day_str in self._daily_cache:
            del self._daily_cache[day_str]
            if day_str in self._cache_order:
                self._cache_order.remove(day_str)
            logger.debug("Removed day %s from cache", day_str)
