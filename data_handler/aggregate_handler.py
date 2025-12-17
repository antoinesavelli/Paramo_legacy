
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
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Dict
import logging
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger("data_handler.aggregate_handler")


class AggregateDataHandler:
    """Handles daily aggregate OHLCV data with efficient caching."""
    
    def __init__(self, aggregate_dir: str, cache_limit: int = 3):
        """
        Initialize aggregate data handler.
        
        Args:
            aggregate_dir: Base directory for aggregates (e.g., D:\trading_data\daily_aggregates)
            cache_limit: Number of months to keep in cache
        """
        self.aggregate_dir = Path(aggregate_dir)
        self._cache = OrderedDict()
        self._cache_limit = cache_limit
        
        logger.info(f"AggregateDataHandler initialized: {self.aggregate_dir}")
        logger.info(f"Cache limit: {cache_limit} months")
    
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
            'volume': int(row['volume']),
            'bar_count': int(row['bar_count']),
            'first_timestamp': row.get('first_timestamp'),
            'last_timestamp': row.get('last_timestamp')
        }
        
        # Add volume averages if available
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
        Load monthly aggregate file with LRU caching.
        
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
        
        # Load from file
        year = year_month[:4]
        agg_file = self.aggregate_dir / year / f"{year_month}.parquet"
        
        if not agg_file.exists():
            logger.debug(f"No aggregate file for {year_month}: {agg_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(agg_file)
            
            # Ensure date column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # LRU cache eviction
            if len(self._cache) >= self._cache_limit:
                evicted = self._cache.popitem(last=False)
                logger.debug(f"Evicted from cache: {evicted[0]}")
            
            self._cache[year_month] = df
            logger.info(f"Loaded aggregate: {year_month} ({len(df)} rows)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading aggregate file {agg_file}: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear the aggregate cache."""
        self._cache.clear()
        logger.info("Cleared aggregate cache")
    
    def cleanup_month(self, year_month: str):
        """
        Remove specific month from cache.
        
        Args:
            year_month: Month in YYYYMM format
        """
        if year_month in self._cache:
            del self._cache[year_month]
            logger.debug(f"Removed {year_month} from cache")
    
    def get_cache_info(self) -> Dict:
        """
        Get information about current cache state.
        
        Returns:
            Dict with cache statistics
        """
        return {
            'cached_months': list(self._cache.keys()),
            'cache_size': len(self._cache),
            'cache_limit': self._cache_limit,
            'total_rows': sum(len(df) for df in self._cache.values())
        }