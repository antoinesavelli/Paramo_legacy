# =====================================================
# data_handler.local.py - Market Data Management (Local/Backtest)
# =====================================================

import pandas as pd
from datetime import datetime
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

    def __init__(self, config, data_dir=r"D:\trading_data"):
        self.config = config
        self.data_dir = Path(data_dir)
        self._day_cache = OrderedDict()
        self._day_cache_limit = 100
        self._symbol_cache = OrderedDict()
        self._symbol_cache_limit = 500
        self._filtered_cache = OrderedDict()
        self._missing_days_cache = set()
        
        # Build index for hierarchical structure
        self._file_index = self._build_file_index()
        logger.info(f"Indexed {len(self._file_index)} dates with {sum(len(v) for v in self._file_index.values())} total symbol files")
        
        # Initialize gap calculator
        self._gap_calculator = GapCalculator(
            get_day_df_func=self._get_day_df,
            get_symbol_day_data_func=self.get_symbol_day_data,
            file_index=self._file_index,
            cache_limit=50
        )
    
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
        """Build index: {date_str: {symbol1, symbol2, ...}} for hierarchical YYYY/MM/DD structure."""
        index = {}
        pattern = re.compile(r"^([A-Z0-9]+)_(\d{8})\.parquet$", re.IGNORECASE)
        
        try:
            if not self.data_dir.exists():
                logger.warning(f"Data directory does not exist: {self.data_dir}")
                return index
            
            for year_dir in self.data_dir.iterdir():
                if not year_dir.is_dir() or not year_dir.name.isdigit():
                    continue
                    
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir() or not month_dir.name.isdigit():
                        continue
                    
                    for day_dir in month_dir.iterdir():
                        if not day_dir.is_dir() or not day_dir.name.isdigit():
                            continue
                        
                        date_str = f"{year_dir.name}-{month_dir.name}-{day_dir.name}"
                        
                        for file in day_dir.glob("*.parquet"):
                            match = pattern.match(file.name)
                            if match:
                                symbol = match.group(1).upper()
                                
                                if date_str not in index:
                                    index[date_str] = set()
                                index[date_str].add(symbol)
            
            logger.info(f"Indexed {len(index)} unique dates")
            
        except Exception as e:
            logger.error(f"Error building file index: {e}", exc_info=True)
        
        return index
    
    def _get_symbol_date_file(self, symbol: str, date_str: str) -> Optional[Path]:
        """Check file index before constructing path."""
        if date_str not in self._file_index:
            return None
        
        if symbol not in self._file_index[date_str]:
            return None
        
        try:
            date_obj = pd.to_datetime(date_str)
            year = str(date_obj.year)
            month = f"{date_obj.month:02d}"
            day = f"{date_obj.day:02d}"
            
            date_compact = date_str.replace('-', '')
            filename = f"{symbol}_{date_compact}.parquet"
            filepath = self.data_dir / year / month / day / filename
            
            return filepath if filepath.exists() else None
            
        except Exception as e:
            logger.debug(f"Error constructing file path for {symbol} on {date_str}: {e}")
            return None
    
    def _get_day_df(self, date_str: str) -> pd.DataFrame:
        """Load all symbols for a specific date using hierarchical file structure."""
        if date_str in self._day_cache:
            self._day_cache.move_to_end(date_str)
            return self._day_cache[date_str]
        
        if date_str in self._missing_days_cache:
            return pd.DataFrame()
        
        if date_str not in self._file_index:
            self._missing_days_cache.add(date_str)
            return pd.DataFrame()
        
        symbols = self._file_index[date_str]
        
        if not symbols:
            self._missing_days_cache.add(date_str)
            return pd.DataFrame()
        
        parts = []
        
        for symbol in symbols:
            file_path = self._get_symbol_date_file(symbol, date_str)
            if file_path:
                symbol_df = self._read_parquet_safe(file_path)
                if not symbol_df.empty:
                    if 'symbol' not in symbol_df.columns:
                        symbol_df['symbol'] = symbol
                    parts.append(symbol_df)
        
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        
        if not df.empty:
            df = self._normalize_columns(df)
            if len(self._day_cache) >= self._day_cache_limit:
                self._day_cache.popitem(last=False)
            self._day_cache[date_str] = df
        else:
            self._missing_days_cache.add(date_str)
        
        return df

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
            
            symbol_file = self._get_symbol_date_file(symbol, date_str)
            
            if symbol_file:
                sdf = self._read_parquet_safe(symbol_file)
                if not sdf.empty:
                    sdf = self._normalize_columns(sdf)
                    parts.append(sdf)
            
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
                        return out.rename(columns={col: 'symbol'})
        except Exception as e:
            logger.debug(f"Could not load universe file: {e}")
    
        symbols = set()
        for date_symbols in self._file_index.values():
            symbols.update(date_symbols)
    
        return pd.DataFrame({'symbol': sorted(symbols)}) if symbols else pd.DataFrame(columns=['symbol'])

    def get_intraday_bars(self, symbol: str, timeframe: str = '1Min',
                          start: Optional[datetime] = None,
                          end: Optional[datetime] = None,
                          limit: Optional[int] = None) -> pd.DataFrame:
        """Get intraday bars for a symbol with early-exit for missing data."""
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
            
            symbol_file = self._get_symbol_date_file(symbol, date_str)
            
            if symbol_file:
                sdf = self._read_parquet_safe(symbol_file)
                if not sdf.empty:
                    sdf = self._normalize_columns(sdf)
                    parts.append(sdf)
            
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

        # Keep timestamps in UTC for consistency across all data operations
        # Timezone conversions should only happen at presentation layer if needed
        return df_all[['timestamp', 'open', 'high', 'low', 'close', 'volume']] if not df_all.empty else pd.DataFrame()

    def calculate_gaps(self, day: datetime, premarket: bool = False) -> Dict:
        """Delegate to GapCalculator."""
        return self._gap_calculator.calculate_gaps(day, premarket)

    def calculate_gaps_efficient(self, day: datetime, premarket: bool = False) -> Dict:
        """Delegate to GapCalculator."""
        return self._gap_calculator.calculate_gaps_efficient(day, premarket)

    def get_day_df_filtered(self, date_str: str, 
                            min_price: Optional[float] = None,
                            max_price: Optional[float] = None,
                            min_volume: Optional[int] = None) -> pd.DataFrame:
        """Load day data with basic filters applied BEFORE caching."""
        if not self.has_data_for_date(date_str):
            return pd.DataFrame()
        
        cache_key = (date_str, min_price, max_price, min_volume)
        if cache_key in self._filtered_cache:
            self._filtered_cache.move_to_end(cache_key)
            return self._filtered_cache[cache_key]
            
        df = self._get_day_df(date_str)
        if df.empty:
            return df
        
        if min_price is not None:
            df = df[df['close'] >= min_price]
        if max_price is not None:
            df = df[df['close'] <= max_price]
        if min_volume is not None:
            df = df[df['volume'] >= min_volume]
        
        if len(self._filtered_cache) >= 50:
            self._filtered_cache.popitem(last=False)
        self._filtered_cache[cache_key] = df
        
        return df

    def get_symbol_day_data(self, symbol: str, date_str: str) -> pd.DataFrame:
        """Efficiently get data for a specific symbol on a specific day."""
        cache_key = (symbol, date_str)
        if cache_key in self._symbol_cache:
            self._symbol_cache.move_to_end(cache_key)
            return self._symbol_cache[cache_key]
        
        if date_str not in self._file_index or symbol not in self._file_index[date_str]:
            return pd.DataFrame()
        
        symbol_file = self._get_symbol_date_file(symbol, date_str)
        if symbol_file:
            symbol_df = self._read_parquet_safe(symbol_file)
            if not symbol_df.empty:
                symbol_df = self._normalize_columns(symbol_df)
                if len(self._symbol_cache) >= self._symbol_cache_limit:
                    self._symbol_cache.popitem(last=False)
                self._symbol_cache[cache_key] = symbol_df
                return symbol_df
        
        return pd.DataFrame()

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
    
    def _read_parquet_safe(self, filepath: Path) -> pd.DataFrame:
        """Safely read parquet file."""
        try:
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
            # Parse timestamps and localize to configured timezone (US/Eastern)
            # then convert to UTC for consistent internal representation
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # If timestamps are naive (no timezone), assume they're in DATA_TZ (US/Eastern)
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT')

            # Convert to UTC for consistent internal storage
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

        return df