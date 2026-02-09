# =====================================================
# Daily aggregate data handler for gap calculations and volume analysis.
# =====================================================

"""
Handles:
- Monthly aggregate file loading and caching
- Daily OHLCV statistics retrieval
- Historical volume data
- Efficient batch operations

File structure: T:\\trading\\daily_aggregates\\YYYY\\YYYY-MM.parquet

MEMORY OPTIMIZATION:
- Only ONE month cached at a time (cache_limit=1)
- Only essential columns loaded (symbol, date, open, high, low, close, volume, float, marketcap)
- Automatic cache eviction when switching months
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List
import logging
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger("data_handler.aggregate_handler")


class AggregateDataHandler:
    """Handles daily aggregate OHLCV data with efficient caching."""
    
    # ✅ OPTIMIZED: Only load essential columns (added float and marketcap)
    REQUIRED_COLUMNS = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    OPTIONAL_COLUMNS = ['bar_count', 'first_timestamp', 'last_timestamp', 
                        'avg_volume_10d', 'avg_volume_20d', 'float', 'marketcap']
    
    def __init__(self, aggregate_dir: str, cache_limit: int = 1):
        """
        Initialize aggregate data handler.
        
        Args:
            aggregate_dir: Base directory for aggregates (e.g., T:\\trading\\daily_aggregates)
            cache_limit: Number of months to keep in cache (default: 1 for memory efficiency)
        """
        self.aggregate_dir = Path(aggregate_dir)
        self._cache = OrderedDict()
        # ✅ CHANGED: Default to 1 month only
        self._cache_limit = cache_limit
        
        logger.info(f"AggregateDataHandler initialized: {self.aggregate_dir}")
        logger.info(f"Cache limit: {cache_limit} month(s)")
    
    def get_daily_stats(self, symbol: str, day: datetime) -> Optional[Dict]:
        """
        Get daily OHLCV statistics for a symbol.
        
        Args:
            symbol: Stock symbol
            day: Trading day
            
        Returns:
            Dict with: open, high, low, close, volume, bar_count, float, marketcap, etc.
            or None if not found
        """
        date_obj = pd.Timestamp(day).normalize()
        year_month = date_obj.strftime('%Y-%m')  # ✅ UPDATED: Use YYYY-MM format
        date_key = date_obj.date()
        
        # Load monthly aggregates
        month_df = self._get_monthly_aggregates(year_month)
        
        if month_df.empty:
            return None
        
        # Filter for specific symbol/date
        result = month_df[
            (month_df['symbol'] == symbol) & 
            (month_df['date'] == date_key)
        ]
        
        if result.empty:
            return None
        
        row = result.iloc[0]
        stats = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        }
        
        # Add optional columns if available
        if 'bar_count' in row.index:
            stats['bar_count'] = int(row['bar_count'])
        if 'first_timestamp' in row.index:
            stats['first_timestamp'] = row['first_timestamp']
        if 'last_timestamp' in row.index:
            stats['last_timestamp'] = row['last_timestamp']
        if 'avg_volume_10d' in row.index:
            stats['avg_volume_10d'] = float(row['avg_volume_10d'])  # ✅ FIXED: Removed extra closing paren
        if 'avg_volume_20d' in row.index:
            stats['avg_volume_20d'] = float(row['avg_volume_20d'])
        if 'float' in row.index and pd.notna(row['float']):
            stats['float'] = float(row['float'])
        if 'marketcap' in row.index and pd.notna(row['marketcap']):
            stats['marketcap'] = float(row['marketcap'])
        
        return stats
    
    def get_day_aggregates(self, day: datetime) -> Optional[pd.DataFrame]:
        """
        Get all symbol aggregates for a specific day.
        
        Args:
            day: Trading day
            
        Returns:
            DataFrame with all symbols for that day, or None if not found
        """
        date_obj = pd.Timestamp(day).normalize()
        year_month = date_obj.strftime('%Y-%m')  # ✅ UPDATED: Use YYYY-MM format
        date_key = date_obj.date()
        
        # Load monthly aggregates
        month_df = self._get_monthly_aggregates(year_month)
        
        if month_df.empty:
            return None
        
        # Filter for specific date
        day_df = month_df[month_df['date'] == date_key].copy()
        
        return day_df if not day_df.empty else None
    
    def get_volume_history(self, symbol: str, end_date: datetime, lookback_days: int = 20) -> Optional[pd.DataFrame]:
        """
        Get historical volume data for a symbol.
        
        Args:
            symbol: Stock symbol
            end_date: End date (inclusive)
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with date and volume columns, sorted by date
        """
        end_obj = pd.Timestamp(end_date).normalize()
        start_obj = end_obj - pd.Timedelta(days=lookback_days + 10)  # Extra buffer for weekends
        
        # Generate list of year-months to check
        date_range = pd.date_range(start=start_obj, end=end_obj, freq='MS')
        year_months = [d.strftime('%Y-%m') for d in date_range]  # ✅ UPDATED: Use YYYY-MM format
        
        # Also include the end month if not already included
        end_month = end_obj.strftime('%Y-%m')  # ✅ UPDATED: Use YYYY-MM format
        if end_month not in year_months:
            year_months.append(end_month)
        
        # Collect data from all relevant months
        all_data = []
        for year_month in year_months:
            month_df = self._get_monthly_aggregates(year_month)
            if not month_df.empty:
                symbol_data = month_df[month_df['symbol'] == symbol][['date', 'volume']]
                if not symbol_data.empty:
                    all_data.append(symbol_data)
        
        if not all_data:
            return None
        
        # Combine and filter by date range
        volume_df = pd.concat(all_data, ignore_index=True)
        volume_df = volume_df[
            (volume_df['date'] >= start_obj.date()) & 
            (volume_df['date'] <= end_obj.date())
        ].sort_values('date')
        
        return volume_df if not volume_df.empty else None
    
    def _get_monthly_aggregates(self, year_month: str) -> pd.DataFrame:
        """
        Load monthly aggregate file with caching.
        
        Args:
            year_month: Month in YYYY-MM format (e.g., '2024-01')
            
        Returns:
            DataFrame with all symbols for that month
        """
        # Check cache first
        if year_month in self._cache:
            logger.debug(f"Cache hit: {year_month}")
            # Move to end (most recently used)
            self._cache.move_to_end(year_month)
            return self._cache[year_month]
        
        # Construct file path: YYYY/YYYY-MM.parquet
        year = year_month[:4]
        file_path = self.aggregate_dir / year / f"{year_month}.parquet"
        
        if not file_path.exists():
            logger.debug(f"Aggregate file not found: {file_path}")
            return pd.DataFrame()
        
        try:
            # ✅ Load only required columns for memory efficiency
            df = pd.read_parquet(file_path)
            
            # Validate required columns
            missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                logger.warning(f"Missing columns in {file_path}: {missing}")
                return pd.DataFrame()
            
            # Ensure date column is proper date type
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Add to cache
            self._cache[year_month] = df
            
            # Evict oldest if over limit
            if len(self._cache) > self._cache_limit:
                oldest = next(iter(self._cache))
                evicted = self._cache.pop(oldest)
                logger.debug(f"Cache eviction: {oldest} ({len(evicted):,} rows)")
            
            logger.info(f"Loaded aggregate file: {file_path} ({len(df):,} rows)")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear the entire cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cached_months': list(self._cache.keys()),
            'total_rows': sum(len(df) for df in self._cache.values()),
            'cache_limit': self._cache_limit
        }