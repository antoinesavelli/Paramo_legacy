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
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import re
import logging
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger("data_handler.local")


class LocalDataHandler:
    """Handles local market data for backtesting with hierarchical YYYY/MM/DD structure."""

    # ✅ OPTIMIZED: Define essential columns only
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
            data_dir: Base directory for intraday data (D:\trading_data)
        """
        self.config = config
        self.data_dir = Path(data_dir)
        
        # ✅ Day cache - keep only 2 days max for memory efficiency
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
        
        Structure: YYYY/MM/YYYYMMDD.parquet
        
        Returns:
            Dict of {date_str: {symbol1, symbol2, ...}}
        """
        index = {}
        pattern = re.compile(r"^(\d{8})\.parquet$", re.IGNORECASE)
        
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
                            date_compact = match.group(1)
                            date_str = f"{date_compact[0:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
                            
                            try:
                                # ✅ OPTIMIZED: Only load symbol column for indexing
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
        # ✅ OPTIMIZATION: Cache key includes columns to avoid loading unnecessary data
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
            date_compact = date_str.replace('-', '')
            filename = f"{date_compact}.parquet"
            filepath = self.data_dir / year / month / filename
            
            if not filepath.exists():
                self._missing_days_cache.add(date_str)
                return pd.DataFrame()
            
            # ✅ OPTIMIZED: Load only requested columns
            df = self._read_parquet_safe(filepath, columns=columns)
            
            if not df.empty:
                df = self._normalize_columns(df)
                
                # ✅ LRU cache eviction with explicit cleanup
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
    def _to_utc(ts) -> pd.Timestamp:
        """Convert timestamp to UTC."""
        t = pd.Timestamp(ts)
        return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    
    @staticmethod
    def _convert_to_et(df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp column to US/Eastern timezone."""
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
        
        # ✅ OPTIMIZED: Define columns needed for bars
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
                # ✅ OPTIMIZED: Only load symbol column
                uni = pd.read_parquet(uni_path, columns=['symbol'])
                uni = self._normalize_columns(uni)
                if not uni.empty:
                    out = uni[['symbol']].dropna().drop_duplicates().reset_index(drop=True)
                    logger.info(f"Loaded universe from file: {len(out)} symbols")
                    return out
        except Exception as e:
            logger.debug(f"Could not load universe file: {e}")

        # Fallback: build from file index
        symbols = set()
        for date_symbols in self._file_index.values():
            symbols.update(date_symbols)

        df = pd.DataFrame({'symbol': sorted(symbols)}) if symbols else pd.DataFrame(columns=['symbol'])
        logger.info(f"Built universe from file index: {len(df)} symbols")
        return df

    def get_intraday_bars(self, symbol: str, timeframe: str = '1Min',
                      start: Optional[datetime] = None,
                      end: Optional[datetime] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get intraday bars for a symbol.
    
        Args:
            symbol: Stock symbol
            timeframe: Timeframe ('1Min')
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars
        
        Returns:
            DataFrame with timestamp, OHLCV columns (and cumulative_volume if present in data)
        """
        logger.debug(f"Processing intraday bars for {symbol} from {start} to {end}")

        if timeframe != '1Min':
            timeframe = '1Min'

        if start is None or end is None:
            raise ValueError("start and end must be provided")

        start_dt = pd.Timestamp(start)
        if start_dt.hour == 0 and start_dt.minute == 0:
            start_dt = start_dt.replace(hour=4, minute=0)
    
        end_dt = pd.Timestamp(end)
        if end_dt.hour == 0 and end_dt.minute == 0:
            end_dt = end_dt.replace(hour=20, minute=0)

        # ✅ OPTIMIZED: Load only needed columns
        needed_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'cumulative_volume']
        
        cur = start_dt.normalize()
        end_ts = end_dt.normalize()
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
                    symbol_df = self._normalize_columns(symbol_df)
                    # Filter out zero-price bars
                    symbol_df = symbol_df[
                        (symbol_df['open'] > 0) & 
                        (symbol_df['close'] > 0)
                    ].copy()
                    if not symbol_df.empty:
                        parts.append(symbol_df)
    
            cur += pd.Timedelta(days=1)

        if not parts:
            return pd.DataFrame()

        df_all = pd.concat(parts, ignore_index=True)
        df_all = df_all.dropna(subset=['timestamp'])
    
        start_utc = self._to_utc(start_dt)
        end_utc = self._to_utc(end_dt)
        df_all = df_all[(df_all['timestamp'] >= start_utc) & (df_all['timestamp'] <= end_utc)]
        df_all = df_all.sort_values('timestamp')
    
        if limit:
            df_all = df_all.tail(limit)

        # Return columns that exist
        base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if not df_all.empty and 'cumulative_volume' in df_all.columns:
            return_cols = base_cols + ['cumulative_volume']
        else:
            return_cols = base_cols
        
        return df_all[[col for col in return_cols if col in df_all.columns]] if not df_all.empty else pd.DataFrame()

    def get_symbol_day_data(self, symbol: str, date_str: str) -> pd.DataFrame:
        """
        Get data for a specific symbol on a specific day.
    
        Args:
            symbol: Stock symbol
            date_str: Date in YYYY-MM-DD format
        
        Returns:
            DataFrame with symbol's data for that day
        """
        if date_str not in self._file_index or symbol not in self._file_index[date_str]:
            return pd.DataFrame()
    
        day_df = self._get_day_df(date_str)
        if day_df.empty:
            return pd.DataFrame()
    
        symbol_df = day_df[day_df['symbol'] == symbol].copy()
        return symbol_df

    def get_regular_hours_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to regular trading hours (9:30 AM - 4:00 PM ET).
        
        Args:
            df: DataFrame with timestamp or ms_of_day column
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
            
        if 'ms_of_day' not in df.columns:
            if 'timestamp' in df.columns:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df = df.between_time('09:30', '16:00')
                df = df.reset_index()
            return df
        
        # 09:30 = 34200000ms, 16:00 = 57600000ms
        return df[(df['ms_of_day'] >= 34200000) & (df['ms_of_day'] < 57600000)]
    
    def _read_parquet_safe(self, filepath: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Safely read parquet file with optional column filtering.
        
        Args:
            filepath: Path to parquet file
            columns: Optional list of columns to load
        """
        try:
            if columns:
                # ✅ OPTIMIZED: Load only specified columns
                try:
                    # Check which columns actually exist in the file
                    import pyarrow.parquet as pq
                    schema = pq.read_schema(filepath)
                    available_cols = schema.names
                    cols_to_load = [col for col in columns if col in available_cols]
                    
                    if cols_to_load:
                        return pd.read_parquet(filepath, columns=cols_to_load)
                    else:
                        # None of the requested columns exist, load all
                        return pd.read_parquet(filepath)
                        
                except ImportError:
                    # PyArrow not available, try loading with pandas
                    return pd.read_parquet(filepath, columns=columns)
            else:
                return pd.read_parquet(filepath)
                
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return pd.DataFrame()
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and ensure timestamp is UTC."""
        if df.empty:
            return df

        df = df.copy()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT')

            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

        return df
    
    def clear_cache(self):
        """Clear all caches and free memory."""
        days_cleared = len(self._day_cache)
        total_rows = sum(len(df) for df in self._day_cache.values())
        self._day_cache.clear()
        logger.info(f"Cleared day cache: {days_cleared} day(s), {total_rows:,} rows freed")
    
    def get_cache_info(self) -> Dict:
        """Get information about current cache state."""
        return {
            'cached_days': list(self._day_cache.keys()),
            'cache_size': len(self._day_cache),
            'cache_limit': self._day_cache_limit,
            'total_rows': sum(len(df) for df in self._day_cache.values()),
            'missing_days_cached': len(self._missing_days_cache)
        }
    
    def diagnose_data_files(self) -> Dict:
        """
        Diagnostic method to check all parquet files for corruption.
        
        Returns:
            Dict with diagnostic results
        """
        pattern = re.compile(r"^(\d{8})\.parquet$", re.IGNORECASE)
        
        report = {
            'total_files': 0,
            'valid_files': 0,
            'corrupted_files': [],
            'missing_symbol_column': [],
            'empty_files': [],
            'valid_dates': []
        }
        
        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return report
        
        logger.info("=" * 80)
        logger.info("DIAGNOSING DATA FILES")
        logger.info("=" * 80)
        
        for year_dir in self.data_dir.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit() or len(year_dir.name) != 4:
                continue
                
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir() or not month_dir.name.isdigit() or len(month_dir.name) != 2:
                    continue
                
                for file in month_dir.glob("*.parquet"):
                    match = pattern.match(file.name)
                    if not match:
                        continue
                    
                    report['total_files'] += 1
                    date_compact = match.group(1)
                    date_str = f"{date_compact[0:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
                    
                    try:
                        # ✅ OPTIMIZED: Only load symbol column for diagnostics
                        df = pd.read_parquet(file, columns=['symbol'])
                        
                        if df.empty:
                            report['empty_files'].append({
                                'date': date_str,
                                'path': str(file),
                                'reason': 'Empty DataFrame'
                            })
                            logger.warning(f"Empty: {date_str}")
                            continue
                        
                        if 'symbol' not in df.columns:
                            report['missing_symbol_column'].append({
                                'date': date_str,
                                'path': str(file),
                                'columns': list(df.columns)
                            })
                            logger.warning(f"Missing symbol column: {date_str}")
                            continue
                        
                        symbols = df['symbol'].unique()
                        report['valid_files'] += 1
                        report['valid_dates'].append({
                            'date': date_str,
                            'path': str(file),
                            'symbols': len(symbols),
                            'rows': len(df)
                        })
                        logger.info(f"Valid: {date_str} ({len(symbols)} symbols, {len(df)} rows)")
                        
                    except Exception as e:
                        report['corrupted_files'].append({
                            'date': date_str,
                            'path': str(file),
                            'error': str(e)
                        })
                        logger.error(f"Corrupted: {date_str} - {e}")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total files: {report['total_files']}")
        logger.info(f"Valid files: {report['valid_files']}")
        logger.info(f"Corrupted: {len(report['corrupted_files'])}")
        logger.info(f"Empty: {len(report['empty_files'])}")
        logger.info(f"Missing symbol: {len(report['missing_symbol_column'])}")
        logger.info("=" * 80)
        
        return report