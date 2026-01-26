# =====================================================
# Daily aggregate data handler for gap calculations and volume analysis.
# =====================================================

"""
Handles:
- Monthly aggregate file loading and caching
- Daily OHLCV statistics retrieval
- Historical volume data
- Efficient batch operations

File structure: daily_aggregates/YYYY/YYYYMM.parquet

MEMORY OPTIMIZATION:
- Only ONE month cached at a time (cache_limit=1)
- Only essential columns loaded (symbol, date, open, high, low, close, volume)
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
    
    # ✅ OPTIMIZED: Only load essential columns
    REQUIRED_COLUMNS = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    OPTIONAL_COLUMNS = ['bar_count', 'first_timestamp', 'last_timestamp', 
                        'avg_volume_10d', 'avg_volume_20d']
    
    def __init__(self, aggregate_dir: str, cache_limit: int = 1):
        """
        Initialize aggregate data handler.
        
        Args:
            aggregate_dir: Base directory for aggregates (e.g., D:\trading_data\daily_aggregates)
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
            Dict with: open, high, low, close, volume, bar_count, etc.
            or None if not found
        """
        date_obj = pd.Timestamp(day).normalize()
        year_month = date_obj.strftime('%Y%m')
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
            stats['avg_volume_10d'] = float(row['avg_volume_10d'])
        if 'avg_volume_20d' in row.index:
            stats['avg_volume_20d'] = float(row['avg_volume_20d'])
        
        return stats
    
    def get_daily_volume_history(self, symbol: str, start_date: datetime, 
                                  end_date: datetime, bars: int = 20) -> Optional[pd.DataFrame]:
        """
        Get historical daily volume for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            bars: Number of bars to retrieve
            
        Returns:
            DataFrame with date and volume columns, or None
        """
        try:
            cur = pd.Timestamp(end_date).normalize()
            start = pd.Timestamp(start_date).normalize()
            
            volumes = []
            days_collected = 0
            
            while days_collected < bars and cur >= start:
                year_month = cur.strftime('%Y%m')
                
                # Load entire month at once
                month_df = self._get_monthly_aggregates(year_month)
                
                if not month_df.empty:
                    # Filter for this symbol
                    symbol_data = month_df[month_df['symbol'] == symbol].copy()
                    
                    if not symbol_data.empty:
                        # Filter by date range
                        symbol_data = symbol_data[
                            (symbol_data['date'] >= start.date()) &
                            (symbol_data['date'] <= cur.date())
                        ]
                        
                        # Extract volumes
                        for _, row in symbol_data.iterrows():
                            volumes.append({
                                'date': row['date'],
                                'volume': row['volume']
                            })
                            days_collected += 1
                            
                            if days_collected >= bars:
                                break
                
                # Move to previous month
                cur = cur - pd.DateOffset(months=1)
            
            if not volumes:
                return None
            
            df = pd.DataFrame(volumes)
            df = df.sort_values('date')
            return df.tail(bars)
            
        except Exception as e:
            logger.debug(f"Error getting daily volume for {symbol}: {e}")
            return None
    
    def get_day_aggregates(self, day: datetime) -> Optional[pd.DataFrame]:
        """
        Get all symbol aggregates for a specific day.
        
        Args:
            day: Trading day
            
        Returns:
            DataFrame with all symbols' OHLCV data for that day
        """
        date_obj = pd.Timestamp(day).normalize()
        year_month = date_obj.strftime('%Y%m')
        date_key = date_obj.date()
        
        month_df = self._get_monthly_aggregates(year_month)
        
        if month_df.empty:
            return None
        
        day_df = month_df[month_df['date'] == date_key].copy()
        
        return day_df if not day_df.empty else None
    
    def _get_monthly_aggregates(self, year_month: str) -> pd.DataFrame:
        """
        Load monthly aggregate file with single-month caching.
        
        OPTIMIZATION: Only one month is kept in cache at a time.
        When switching months, old month is automatically evicted.
        
        Args:
            year_month: YYYYMM format (e.g., '202401')
            
        Returns:
            DataFrame with daily aggregates for all symbols in that month
        """
        # Check cache first
        if year_month in self._cache:
            self._cache.move_to_end(year_month)
            logger.debug(f"Cache hit: {year_month}")
            return self._cache[year_month]
        
        # ✅ OPTIMIZATION: Evict old month BEFORE loading new one
        if len(self._cache) >= self._cache_limit:
            evicted_month, evicted_df = self._cache.popitem(last=False)
            rows_freed = len(evicted_df)
            del evicted_df  # Explicit cleanup
            logger.info(f"Evicted from cache: {evicted_month} ({rows_freed:,} rows freed)")
        
        # Load from file
        year = year_month[:4]
        agg_file = self.aggregate_dir / year / f"{year_month}.parquet"
        
        if not agg_file.exists():
            logger.debug(f"No aggregate file for {year_month}: {agg_file}")
            return pd.DataFrame()
        
        try:
            # ✅ OPTIMIZED: Only load required columns
            # First check what columns are available
            try:
                # Fast column check without loading data
                import pyarrow.parquet as pq
                schema = pq.read_schema(agg_file)
                available_cols = schema.names
                
                # Build column list: required + available optional
                cols_to_load = [col for col in self.REQUIRED_COLUMNS if col in available_cols]
                for opt_col in self.OPTIONAL_COLUMNS:
                    if opt_col in available_cols:
                        cols_to_load.append(opt_col)
                
                df = pd.read_parquet(agg_file, columns=cols_to_load)
                logger.debug(f"Loaded {len(cols_to_load)} columns from {year_month}")
                
            except ImportError:
                # Fallback if pyarrow not available - load all columns
                df = pd.read_parquet(agg_file)
                # Filter to only needed columns
                available_cols = df.columns.tolist()
                keep_cols = [col for col in (self.REQUIRED_COLUMNS + self.OPTIONAL_COLUMNS) 
                           if col in available_cols]
                df = df[keep_cols]
            
            # Ensure date column is date type (not datetime)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Add to cache
            self._cache[year_month] = df
            
            # Log memory usage estimate
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(f"Loaded aggregate: {year_month} ({len(df):,} rows, {memory_mb:.1f} MB)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading aggregate file {agg_file}: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear the aggregate cache and free memory."""
        months_cleared = list(self._cache.keys())
        total_rows = sum(len(df) for df in self._cache.values())
        self._cache.clear()
        logger.info(f"Cleared aggregate cache: {len(months_cleared)} month(s), {total_rows:,} rows freed")
    
    def cleanup_month(self, year_month: str):
        """
        Remove specific month from cache.
        
        Args:
            year_month: Month in YYYYMM format
        """
        if year_month in self._cache:
            rows = len(self._cache[year_month])
            del self._cache[year_month]
            logger.debug(f"Removed {year_month} from cache ({rows:,} rows)")
    
    def get_cache_info(self) -> Dict:
        """
        Get information about current cache state.
        
        Returns:
            Dict with cache statistics
        """
        cache_info = {
            'cached_months': list(self._cache.keys()),
            'cache_size': len(self._cache),
            'cache_limit': self._cache_limit,
            'total_rows': sum(len(df) for df in self._cache.values())
        }
        
        # Add memory usage estimate
        if self._cache:
            total_memory_mb = sum(
                df.memory_usage(deep=True).sum() / (1024 * 1024) 
                for df in self._cache.values()
            )
            cache_info['memory_mb'] = round(total_memory_mb, 2)
        else:
            cache_info['memory_mb'] = 0
        
        return cache_info