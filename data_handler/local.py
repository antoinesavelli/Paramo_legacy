# =====================================================
# data_handler.local.py - Market Data Management (Local/Backtest)
# =====================================================

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
import os
import re
import logging
from pathlib import Path
from collections import OrderedDict

from data_handler.gap_calculator import GapCalculator

logger = logging.getLogger("data_handler.local")

class LocalDataHandler:
    """Handles local market data for backtesting with hierarchical YYYY/MM/DD structure."""

    REQUIRED_COLUMNS = [
        "symbol", "timestamp", "open", "high", "low", "close", "volume"
    ]
    OPTIONAL_COLUMNS = ["count", "vwap", "ms_of_day", "date"]
    COLUMN_ORDER = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "count", "vwap"]

    def __init__(self, config, data_dir: str = None):
        self.config = config
        self.data_dir = Path(data_dir)
        
        # Day cache - only cache mechanism needed
        self._day_cache = OrderedDict()
        self._day_cache_limit = getattr(
            self.config.system, 
            'MAX_DAY_CACHE_SIZE', 
            2  # Default: keep only 2 days in memory
        )
        
        self._missing_days_cache = set()
        
        # Build index for hierarchical structure
        self._file_index = self._build_file_index()
        logger.info(
            f"Indexed {len(self._file_index)} dates with "
            f"{sum(len(v) for v in self._file_index.values())} total symbol files"
        )
        
        # Daily aggregate cache directory
        self.daily_agg_dir = self.data_dir / "daily_aggregates"
        self._daily_agg_cache = OrderedDict()  # Cache monthly aggregate files
        self._daily_agg_cache_limit = 3  # Keep 3 months in memory (~1-2 MB each)
        
        logger.info(f"Daily aggregates dir: {self.daily_agg_dir}")
        
        # ✅ Initialize GapCalculator with aggregate function
        self.gap_calculator = GapCalculator(
            get_daily_stats_func=self.get_daily_stats,  # ✅ Uses aggregate files
            file_index=self._file_index  # ✅ For backtest data availability checking
        )
        
        logger.info(f"Cache: {self._day_cache_limit} days max (symbols and gaps not cached)")
    
    def has_data_for_date(self, date_str: str) -> bool:
        """
        Fast check if data exists for a date without loading it.
        Uses file index for O(1) lookup.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            True if data exists for this date, False otherwise
        """
        if date_str in self._missing_days_cache:
            return False
        
        has_data = date_str in self._file_index and len(self._file_index[date_str]) > 0
        
        if not has_data:
            self._missing_days_cache.add(date_str)
        
        return has_data
    
    def _build_file_index(self) -> Dict[str, set]:
        """Build index: {date_str: {symbol1, symbol2, ...}} for YYYY/MM/YYYYMMDD.parquet structure."""
        index = {}
        pattern = re.compile(r"^(\d{8})\.parquet$", re.IGNORECASE)
        
        # Track statistics
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
                    
                    # Files are directly in the month directory: YYYY/MM/YYYYMMDD.parquet
                    for file in month_dir.glob("*.parquet"):
                        match = pattern.match(file.name)
                        if match:
                            total_files += 1
                            date_compact = match.group(1)  # YYYYMMDD
                            
                            # Convert YYYYMMDD to YYYY-MM-DD
                            date_str = f"{date_compact[0:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
                            
                            # Read the parquet file to get the list of symbols
                            try:
                                df = pd.read_parquet(file)
                                if 'symbol' in df.columns:
                                    # Filter out None/NaN values from symbols
                                    symbols = set(df['symbol'].dropna().unique())
                                    # Further filter to ensure only string values
                                    symbols = {s for s in symbols if s is not None and isinstance(s, str)}
                                    
                                    if symbols:  # Only add if we have valid symbols
                                        if date_str not in index:
                                            index[date_str] = set()
                                        index[date_str].update(symbols)
                                        successful_files += 1
                                        logger.debug(f"Indexed {len(symbols)} symbols for {date_str}")
                                    else:
                                        logger.warning(f"File {file} has no valid symbols - skipping")
                                        corrupted_files += 1
                                else:
                                    logger.warning(f"File {file} missing 'symbol' column - skipping")
                                    corrupted_files += 1
                            except Exception as e:
                                logger.error(f"Error reading {file}: {e}")
                                corrupted_files += 1
            
            # Log summary statistics
            logger.info(f"File indexing complete:")
            logger.info(f"  • Total parquet files found: {total_files}")
            logger.info(f"  • Successfully indexed: {successful_files}")
            logger.info(f"  • Corrupted/skipped: {corrupted_files}")
            logger.info(f"  • Unique dates indexed: {len(index)}")
            
            # Warn if all files are corrupted
            if total_files > 0 and successful_files == 0:
                logger.error("⚠️  WARNING: All parquet files appear to be corrupted or invalid!")
                logger.error("⚠️  Please verify your data files are valid parquet format.")
            
        except Exception as e:
            logger.error(f"Error building file index: {e}", exc_info=True)
        
        return index
    
    def diagnose_data_files(self) -> Dict:
        """
        Diagnostic method to check all parquet files for corruption.
        Returns a report of file status.
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
                        # Try to read the file
                        df = pd.read_parquet(file)
                        
                        # Check if empty
                        if df.empty:
                            report['empty_files'].append({
                                'date': date_str,
                                'path': str(file),
                                'reason': 'Empty DataFrame'
                            })
                            logger.warning(f"❌ {date_str}: Empty file - {file}")
                            continue
                        
                        # Check for symbol column
                        if 'symbol' not in df.columns:
                            report['missing_symbol_column'].append({
                                'date': date_str,
                                'path': str(file),
                                'columns': list(df.columns)
                            })
                            logger.warning(f"❌ {date_str}: Missing 'symbol' column - {file}")
                            logger.warning(f"   Available columns: {list(df.columns)}")
                            continue
                        
                        # File is valid
                        symbols = df['symbol'].unique()
                        report['valid_files'] += 1
                        report['valid_dates'].append({
                            'date': date_str,
                            'path': str(file),
                            'symbols': len(symbols),
                            'rows': len(df)
                        })
                        logger.info(f"✓ {date_str}: {len(symbols)} symbols, {len(df)} rows")
                        
                    except Exception as e:
                        report['corrupted_files'].append({
                            'date': date_str,
                            'path': str(file),
                            'error': str(e)
                        })
                        logger.error(f"❌ {date_str}: CORRUPTED - {e}")
        
        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total files scanned: {report['total_files']}")
        logger.info(f"Valid files: {report['valid_files']}")
        logger.info(f"Corrupted files: {len(report['corrupted_files'])}")
        logger.info(f"Empty files: {len(report['empty_files'])}")
        logger.info(f"Missing 'symbol' column: {len(report['missing_symbol_column'])}")
        
        if report['corrupted_files']:
            logger.info("")
            logger.info("CORRUPTED FILES:")
            for item in report['corrupted_files'][:10]:  # Show first 10
                logger.info(f"  • {item['date']}: {item['error']}")
            if len(report['corrupted_files']) > 10:
                logger.info(f"  ... and {len(report['corrupted_files']) - 10} more")
        
        logger.info("=" * 80)
        
        return report
    
    def _get_symbol_date_file(self, symbol: str, date_str: str) -> Optional[Path]:
        """Get the daily file path for YYYY/MM/YYYYMMDD.parquet structure."""
        if date_str not in self._file_index:
            return None
        
        if symbol not in self._file_index[date_str]:
            return None
        
        try:
            date_obj = pd.to_datetime(date_str)
            year = str(date_obj.year)
            month = f"{date_obj.month:02d}"
            
            date_compact = date_str.replace('-', '')
            filename = f"{date_compact}.parquet"
            filepath = self.data_dir / year / month / filename
            
            return filepath if filepath.exists() else None
            
        except Exception as e:
            logger.debug(f"Error constructing file path for {date_str}: {e}")
            return None
    
    def _get_day_df(self, date_str: str) -> pd.DataFrame:
        """Load all symbols for a specific date from a single daily file."""
        if date_str in self._day_cache:
            self._day_cache.move_to_end(date_str)
            return self._day_cache[date_str]

        if date_str in self._missing_days_cache:
            return pd.DataFrame()

        if date_str not in self._file_index:
            # ✅ FIX: Only cache as missing if file truly doesn't exist
            self._missing_days_cache.add(date_str)
            logger.debug(f"{date_str}: No data in file index")
            return pd.DataFrame()

        try:
            date_obj = pd.to_datetime(date_str)
            year = str(date_obj.year)
            month = f"{date_obj.month:02d}"
            date_compact = date_str.replace('-', '')
            filename = f"{date_compact}.parquet"
            filepath = self.data_dir / year / month / filename

            if not filepath.exists():
                # ✅ FIX: Only cache as missing if file doesn't exist
                self._missing_days_cache.add(date_str)
                logger.debug(f"{date_str}: File not found at {filepath}")
                return pd.DataFrame()

            df = self._read_parquet_safe(filepath, date_str)

            if not df.empty:
                df = self._normalize_columns(df)
                # Cache eviction - remove oldest if at limit
                if len(self._day_cache) >= self._day_cache_limit:
                    self._day_cache.popitem(last=False)
                self._day_cache[date_str] = df
            else:
                # ✅ FIX: Don't cache as missing if file exists but is corrupt/empty
                # This allows retry on next attempt
                logger.error(
                    f"{date_str}: File exists at {filepath} but returned empty data - "
                    f"possible corruption or read error (not caching as missing to allow retry)"
                )

            return df

        except Exception as e:
            # ✅ FIX: Don't cache as missing on exception - allow retry
            logger.error(f"Error loading day data for {date_str}: {e}", exc_info=True)
            logger.warning(f"{date_str}: Not caching as missing to allow retry on next attempt")
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
        """Get bars for a symbol within a date range."""
        logger.debug(f"Loading bars for {symbol} ({timeframe}) from {start} to {end}")

        if start is None or end is None:
            return pd.DataFrame()
        
        cur = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        parts = []
        
        while cur <= end_ts:
            date_str = cur.strftime("%Y-%m-%d")
            
            if not self.has_data_for_date(date_str):
                cur += pd.Timedelta(days=1)
                continue
            
            # Load entire day and filter for symbol
            day_df = self._get_day_df(date_str)
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
        """Returns DataFrame with symbol column using file index."""
        uni_path = self.data_dir / "universe.parquet"

        try:
            if uni_path.exists():
                uni = pd.read_parquet(uni_path)
                uni = self._normalize_columns(uni)
                if not uni.empty:
                    col = 'symbol' if 'symbol' in uni.columns else 'ticker'
                    if col in uni.columns:
                        out = uni[[col]].dropna().drop_duplicates().reset_index(drop=True)
                        logger.info(f"Loaded universe from universe.parquet: {len(out)} symbols")
                        return out.rename(columns={col: 'symbol'})
        except Exception as e:
            logger.debug(f"Could not load universe file: {e}")

        # Fallback: Use file index (all symbols in parquet files - already filtered)
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
        """Get intraday bars for a symbol - reads from cached day data."""
        logger.debug(f"Processing intraday bars for {symbol} from {start} to {end}")

        if timeframe != '1Min':
            timeframe = '1Min'

        if start is None or end is None:
            raise ValueError("start and end must be provided (date or datetime)")

        start_dt = pd.Timestamp(start)
        if start_dt.hour == 0 and start_dt.minute == 0:
            start_dt = start_dt.replace(hour=4, minute=0)
        
        end_dt = pd.Timestamp(end)
        if end_dt.hour == 0 and end_dt.minute == 0:
            end_dt = end_dt.replace(hour=20, minute=0)

        cur = start_dt.normalize()
        end_ts = end_dt.normalize()
        parts = []
        
        while cur <= end_ts:
            date_str = cur.strftime("%Y-%m-%d")
            
            if not self.has_data_for_date(date_str):
                cur += pd.Timedelta(days=1)
                continue
            
            # Load entire day and filter for symbol
            day_df = self._get_day_df(date_str)
            if not day_df.empty and 'symbol' in day_df.columns:
                symbol_df = day_df[day_df['symbol'] == symbol]
                if not symbol_df.empty:
                    original_count = len(symbol_df)
                    symbol_df = self._normalize_columns(symbol_df)

                    # ✅ FIX: Log timestamp normalization issues
                    if 'timestamp' in symbol_df.columns:
                        nat_count = symbol_df['timestamp'].isna().sum()
                        if nat_count > 0:
                            logger.warning(
                                f"{symbol} {date_str}: {nat_count}/{len(symbol_df)} bars have invalid timestamps (NaT)"
                            )

                    # ✅ FIX: Filter out zero-price bars with logging
                    before_filter = len(symbol_df)
                    symbol_df = symbol_df[
                        (symbol_df['open'] > 0) &
                        (symbol_df['close'] > 0)
                    ].copy()
                    after_filter = len(symbol_df)

                    # Log if significant data was dropped
                    if before_filter > 0 and after_filter < before_filter * 0.5:
                        logger.warning(
                            f"{symbol} {date_str}: Zero-price filter removed {before_filter - after_filter}/{before_filter} bars "
                            f"({(before_filter - after_filter)/before_filter*100:.1f}%) - possible data quality issue"
                        )

                    if not symbol_df.empty:
                        parts.append(symbol_df)
                    elif original_count > 0:
                        logger.warning(
                            f"{symbol} {date_str}: All {original_count} bars filtered out (zero prices or invalid timestamps)"
                        )
        
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

        return df_all[['timestamp', 'open', 'high', 'low', 'close', 'volume']] if not df_all.empty else pd.DataFrame()

    def calculate_gaps(self, day: datetime) -> Dict:
        """
        Calculate gaps using unified GapCalculator.
        
        Uses daily aggregates for 100x faster performance.
        """
        return self.gap_calculator.calculate_gaps(day)

    def get_symbol_day_data(self, symbol: str, date_str: str) -> pd.DataFrame:
        """Get data for a specific symbol on a specific day - reads from cached day data."""
        if date_str not in self._file_index or symbol not in self._file_index[date_str]:
            return pd.DataFrame()
        
        # Load the entire day (from cache if available) and filter for symbol
        day_df = self._get_day_df(date_str)
        if day_df.empty:
            return pd.DataFrame()
        
        symbol_df = day_df[day_df['symbol'] == symbol].copy()
        return symbol_df

    def get_regular_hours_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to regular trading hours using ms_of_day."""
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
    
    def _read_parquet_safe(self, filepath: Path, date_str: str = None) -> pd.DataFrame:
        """Safely read parquet file with detailed error reporting."""
        try:
            df = pd.read_parquet(filepath)

            # ✅ FIX: Validate parquet file contents
            if df.empty:
                logger.warning(f"Parquet file is empty: {filepath}")
                return df

            if 'symbol' not in df.columns:
                logger.error(f"Parquet file missing 'symbol' column: {filepath}, columns: {df.columns.tolist()}")
                return pd.DataFrame()

            return df

        except Exception as e:
            # ✅ FIX: More detailed error reporting
            date_info = f" for {date_str}" if date_str else ""
            logger.error(f"Error reading parquet file{date_info}: {filepath}")
            logger.error(f"Error details: {type(e).__name__}: {e}")
            return pd.DataFrame()
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and ensure timestamp is UTC."""
        if df.empty:
            return df

        df = df.copy()

        if 'timestamp' in df.columns:
            original_count = len(df)

            # ✅ FIX: Track timestamp conversion issues
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            nat_from_coerce = df['timestamp'].isna().sum()
            if nat_from_coerce > 0:
                logger.warning(
                    f"Timestamp conversion: {nat_from_coerce}/{original_count} bars have invalid timestamps "
                    f"that couldn't be parsed (set to NaT)"
                )

            if df['timestamp'].dt.tz is None:
                # ✅ FIX: Track timezone localization issues
                before_tz = df['timestamp'].notna().sum()
                df['timestamp'] = df['timestamp'].dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT')
                after_tz = df['timestamp'].notna().sum()

                nat_from_tz = before_tz - after_tz
                if nat_from_tz > 0:
                    logger.warning(
                        f"Timezone localization: {nat_from_tz} bars had ambiguous/nonexistent timestamps "
                        f"(e.g., DST transitions) - set to NaT"
                    )

            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

        return df
    
    def _get_monthly_aggregates(self, year_month: str) -> pd.DataFrame:
        """
        Load monthly aggregate file (fast, cached).
        
        Args:
            year_month: YYYYMM format (e.g., '202401')
            
        Returns:
            DataFrame with daily aggregates for all symbols in that month
        """
        # Check cache first
        if year_month in self._daily_agg_cache:
            self._daily_agg_cache.move_to_end(year_month)
            return self._daily_agg_cache[year_month]
        
        # Load from file
        year = year_month[:4]
        agg_file = self.daily_agg_dir / year / f"{year_month}.parquet"
        
        if not agg_file.exists():
            logger.debug(f"No aggregate file for {year_month}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(agg_file)
            
            # Ensure date column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Cache with eviction
            if len(self._daily_agg_cache) >= self._daily_agg_cache_limit:
                self._daily_agg_cache.popitem(last=False)
            
            self._daily_agg_cache[year_month] = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading aggregate file {agg_file}: {e}")
            return pd.DataFrame()

    def get_daily_stats(self, symbol: str, day: datetime) -> Optional[Dict]:
        """
        Get daily OHLCV statistics from aggregate files.
        
        Returns dict with: 'open', 'high', 'low', 'close', 'volume', 'avg_volume_10d', 'avg_volume_20d'
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
        
        # ✅ NEW: Add volume averages if available
        if 'avg_volume_10d' in row.index:
            stats['avg_volume_10d'] = float(row['avg_volume_10d'])
        if 'avg_volume_20d' in row.index:
            stats['avg_volume_20d'] = float(row['avg_volume_20d'])
        
        return stats

    def get_daily_volume_history(self, symbol: str, start_date: datetime, 
                                 end_date: datetime, bars: int = 20) -> Optional[pd.DataFrame]:
        """Get historical daily volume - NOW USING AGGREGATES (10x faster!)."""
        try:
            # Determine which months to load
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
            return df.tail(bars)  # Return only requested number of bars
            
        except Exception as e:
            logger.debug(f"Error getting daily volume for {symbol}: {e}")
            return None