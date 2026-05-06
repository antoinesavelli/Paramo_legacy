# =====================================================
# data_handler.local.py - Market Data Management (Local/Backtest)
# =====================================================

"""
Local data handler for backtesting with hierarchical file structure.

Handles:
- File indexing (with optional caching)
- Intraday bar loading
- Day-level data caching
- Universe management

MEMORY OPTIMIZATION:
- Only essential columns loaded from parquet files
- Day cache limited to 2 days max
- Explicit cleanup of unused data

FILE FORMAT: YYYY-MM-DD.parquet (ISO date format)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import re
import logging
from pathlib import Path
from collections import OrderedDict
from utils.helpers import validate_ohlcv

logger = logging.getLogger("data_handler.local")


class LocalDataHandler:
    """Handles local market data for backtesting with hierarchical YYYY/MM/DD structure."""

    # NOTE: OPTIMIZED: Define essential columns only
    REQUIRED_COLUMNS = [
        "symbol", "timestamp", "open", "high", "low", "close", "volume"
    ]
    OPTIONAL_COLUMNS = ["count", "vwap", "ms_of_day", "date", "cumulative_volume"]
    COLUMN_ORDER = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "count", "vwap"]

    def __init__(self, config, data_dir: str = None):
        """
        Initialize local data handler.
        
        Args:
            config: TradingConfig object
            data_dir: Base directory for intraday data (S:\\trading\\ticker_data)
        """
        self.config = config
        self.data_dir = Path(data_dir)
        
        # NOTE: Day cache - keep only 2 days max for memory efficiency
        self._day_cache = OrderedDict()
        self._day_cache_limit = getattr(
            self.config.system, 
            'MAX_DAY_CACHE_SIZE', 
            2  # Default: keep only 2 days in memory
        )
        
        self._missing_days_cache = set()
        
        # Build or load file index
        if self.config.system.USE_FILE_INDEX_CACHE:
            self._file_index = self._load_or_build_cached_index()
        else:
            self._file_index = self._build_file_index()
        
        logger.info(
            f"Indexed {len(self._file_index)} dates with "
            f"{sum(len(v) for v in self._file_index.values())} total symbols"
        )
        
        logger.info(f"Cache: {self._day_cache_limit} days max")
    
    def _load_or_build_cached_index(self) -> Dict[str, set]:
        """Load file index from cache or build and cache it."""
        try:
            from data_handler.file_index_cache import create_index_for_backtest
            
            logger.info("Loading file index cache...")
            index_data = create_index_for_backtest(self.config)
            
            return index_data['file_index']
            
        except Exception as e:
            logger.warning(f"Failed to load cached index: {e}")
            logger.info("Falling back to building index from scratch...")
            return self._build_file_index()
    
    def has_data_for_date(self, date_str: str) -> bool:
        """
        Fast check if data exists for a date without loading it.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            True if data exists for this date
        """
        if date_str in self._missing_days_cache:
            return False
        
        has_data = date_str in self._file_index and len(self._file_index[date_str]) > 0
        
        if not has_data:
            self._missing_days_cache.add(date_str)
        
        return has_data
    
    def _build_file_index(self) -> Dict[str, set]:
        """
        Build file index from filesystem.
        
        Structure: YYYY/MM/YYYY-MM-DD.parquet
        
        Returns:
            Dict of {date_str: {symbol1, symbol2, ...}}
        """
        index = {}
        # NOTE: NEW FORMAT: YYYY-MM-DD.parquet
        pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$", re.IGNORECASE)
        
        total_files = 0
        corrupted_files = 0
        successful_files = 0
        
        try:
            if not self.data_dir.exists():
                logger.warning(f"Data directory does not exist: {self.data_dir}")
                return index
            
            for year_dir in self.data_dir.iterdir():
                if not year_dir.is_dir() or not year_dir.name.isdigit() or len(year_dir.name) != 4:
                    continue
                    
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir() or not month_dir.name.isdigit() or len(month_dir.name) != 2:
                        continue
                    
                    for file in month_dir.glob("*.parquet"):
                        match = pattern.match(file.name)
                        if match:
                            total_files += 1
                            date_str = match.group(1)  # Already in YYYY-MM-DD format
                            
                            try:
                                # NOTE: OPTIMIZED: Only load symbol column for indexing
                                df = pd.read_parquet(file, columns=['symbol'])
                                symbols = set(df['symbol'].dropna().unique())
                                symbols = {s for s in symbols if s is not None and isinstance(s, str)}
                                
                                if symbols:
                                    if date_str not in index:
                                        index[date_str] = set()
                                    index[date_str].update(symbols)
                                    successful_files += 1
                                    
                                    if successful_files % 50 == 0:
                                        logger.info(f"  Progress: {successful_files} files indexed...")
                                else:
                                    logger.warning(f"File {file} has no valid symbols")
                                    corrupted_files += 1
                                    
                            except Exception as e:
                                logger.error(f"Error reading {file}: {e}")
                                corrupted_files += 1
            
            logger.info(f"File indexing complete:")
            logger.info(f"  • Total files: {total_files}")
            logger.info(f"  • Successfully indexed: {successful_files}")
            logger.info(f"  • Corrupted/skipped: {corrupted_files}")
            logger.info(f"  • Unique dates: {len(index)}")
            
            if total_files > 0 and successful_files == 0:
                logger.error("⚠️  All parquet files appear corrupted!")
            
        except Exception as e:
            logger.error(f"Error building file index: {e}", exc_info=True)
        
        return index
    
    def _get_day_df(self, date_str: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load all symbols for a specific date.
        
        Uses LRU cache for performance.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            columns: Optional list of columns to load (None = load all)
        """
        # NOTE: OPTIMIZATION: Cache key includes columns to avoid loading unnecessary data
        cache_key = date_str
        
        # Check cache first
        if cache_key in self._day_cache:
            self._day_cache.move_to_end(cache_key)
            cached_df = self._day_cache[cache_key]
            
            # If specific columns requested, filter cached data
            if columns and not cached_df.empty:
                available_cols = [col for col in columns if col in cached_df.columns]
                return cached_df[available_cols] if available_cols else cached_df
            
            return cached_df
        
        if date_str in self._missing_days_cache:
            return pd.DataFrame()
        
        if date_str not in self._file_index:
            self._missing_days_cache.add(date_str)
            return pd.DataFrame()
        
        try:
            date_obj = pd.to_datetime(date_str)
            year = str(date_obj.year)
            month = f"{date_obj.month:02d}"
            # NOTE: NEW FORMAT: Use date_str directly (already YYYY-MM-DD)
            filename = f"{date_str}.parquet"
            filepath = self.data_dir / year / month / filename
            
            if not filepath.exists():
                self._missing_days_cache.add(date_str)
                return pd.DataFrame()
            
            # NOTE: OPTIMIZED: Load only requested columns
            df = self._read_parquet_safe(filepath, columns=columns)
            
            if not df.empty:
                df = self._normalize_columns(df)
                df = validate_ohlcv(df, source=date_str, logger=logger)
                
                # NOTE: LRU cache eviction with explicit cleanup
                if len(self._day_cache) >= self._day_cache_limit:
                    evicted_key, evicted_df = self._day_cache.popitem(last=False)
                    rows_freed = len(evicted_df)
                    del evicted_df  # Explicit cleanup
                    logger.debug(f"Evicted day cache: {evicted_key} ({rows_freed:,} rows)")
                
                self._day_cache[cache_key] = df
            else:
                self._missing_days_cache.add(date_str)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading day data for {date_str}: {e}")
            self._missing_days_cache.add(date_str)
            return pd.DataFrame()
    
    @staticmethod
    def _read_parquet_safe(filepath: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Safely read parquet with column filtering."""
        try:
            if columns:
                return pd.read_parquet(filepath, columns=columns)
            else:
                return pd.read_parquet(filepath)
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        if df.empty:
            return df
        df.columns = df.columns.str.lower()
        return df
    
    @staticmethod
    def _to_utc(dt):
        """Convert datetime to UTC."""
        if dt is None:
            return None
        dt = pd.Timestamp(dt)
        if dt.tz is None:
            return dt.tz_localize('UTC')
        return dt.tz_convert('UTC')
    
    @staticmethod
    def _convert_to_et(df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp column to Eastern Time."""
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')
        
        return df

    def get_bars(self, symbol, timeframe, start=None, end=None, limit=None):
        """
        Get bars for a symbol within a date range.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe (currently only '1Min' supported)
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars to return
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.debug(f"Loading bars for {symbol} ({timeframe}) from {start} to {end}")

        if start is None or end is None:
            return pd.DataFrame()
        
        # NOTE: OPTIMIZED: Define columns needed for bars
        needed_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        cur = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        parts = []
        
        while cur <= end_ts:
            date_str = cur.strftime("%Y-%m-%d")
            
            if not self.has_data_for_date(date_str):
                cur += pd.Timedelta(days=1)
                continue
            
            day_df = self._get_day_df(date_str, columns=needed_cols)
            if not day_df.empty and 'symbol' in day_df.columns:
                symbol_df = day_df[day_df['symbol'] == symbol]
                if not symbol_df.empty:
                    parts.append(symbol_df)
            
            cur += pd.Timedelta(days=1)
        
        if not parts:
            return pd.DataFrame()
        
        df = pd.concat(parts, ignore_index=True)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

            if start:
                start_utc = self._to_utc(start)
                df = df[df['timestamp'] >= start_utc]
            if end:
                end_utc = self._to_utc(end)
                df = df[df['timestamp'] <= end_utc]

            df = df.sort_values('timestamp')

        if limit:
            df = df.tail(limit)

        return df

    def get_universe(self) -> pd.DataFrame:
        """
        Get universe of available symbols.
        
        Returns:
            DataFrame with 'symbol' column
        """
        uni_path = self.data_dir / "universe.parquet"

        try:
            if uni_path.exists():
                # NOTE: OPTIMIZED: Only load symbol column
                uni = pd.read_parquet(uni_path, columns=['symbol'])
                uni = self._normalize_columns(uni)
                if not uni.empty:
                    out = uni[['symbol']].dropna().drop_duplicates().reset_index(drop=True)
                    logger.info(f"Loaded universe from file: {len(out)} symbols")
                    return out
        except Exception as e:
            logger.warning(f"Could not load universe file: {e}")

        if self._file_index:
            all_symbols = set()
            for sym_set in self._file_index.values():
                all_symbols.update(sym_set)
            
            if all_symbols:
                df = pd.DataFrame({'symbol': sorted(all_symbols)})
                logger.info(f"Built universe from file index: {len(df)} symbols")
                return df

        logger.warning("No universe data available")
        return pd.DataFrame(columns=['symbol'])

    def clear_cache(self):
        """Clear all cached data to free memory."""
        cache_size = len(self._day_cache)
        self._day_cache.clear()
        logger.info(f"Cleared day cache ({cache_size} entries)")
    
    def get_intraday_bars(self, symbol: str, start=None, end=None, timeframe: str = '1Min') -> pd.DataFrame:
        """Get intraday bar data for pattern analysis. Wraps get_bars for screener compatibility."""
        return self.get_bars(symbol, timeframe, start=start, end=end)