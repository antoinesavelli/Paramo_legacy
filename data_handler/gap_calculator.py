# =====================================================
# data_handler.gap_calculator.py - Gap calculation for market data analysis
# =====================================================

import pandas as pd
from datetime import datetime
from typing import Dict, Callable
from collections import OrderedDict
import logging

logger = logging.getLogger("data_handler.gap_calculator")


class GapCalculator:
    """Handles gap calculations between trading days."""
    
    def __init__(self, get_day_df_func: Callable, get_symbol_day_data_func: Callable, 
                 file_index: Dict, cache_limit: int = 50):
        """
        Args:
            get_day_df_func: Function to get full day data
            get_symbol_day_data_func: Function to get symbol-specific day data
            file_index: File index for checking data availability
            cache_limit: Maximum cache size
        """
        self._get_day_df = get_day_df_func
        self._get_symbol_day_data = get_symbol_day_data_func
        self._file_index = file_index
        self._gap_cache = OrderedDict()
        self._gap_cache_limit = cache_limit
    
    def calculate_gaps(self, day: datetime, premarket: bool = False) -> Dict:
        """
        Compute open gap vs prior day's close for all symbols present in both days.
        Automatically searches for most recent trading day if immediate previous day has no data.
        
        Args:
            day: Trading day to calculate gaps for
            premarket: If True, use premarket open (4:00 AM ET). If False, use regular open (9:30 AM ET)
        """
        day = pd.Timestamp(day).normalize()
        today_str = day.strftime("%Y-%m-%d")
        
        # ✅ FAST PATH: Check if today has data using file index (no I/O)
        if today_str not in self._file_index or len(self._file_index[today_str]) == 0:
            return {
                'gaps': pd.DataFrame(columns=['symbol', 'open_price', 'prev_close', 'last_price', 'gap_percent']),
                'prev_syms': set(),
                'today_syms': set(),
                'prev_empty': True,
                'today_empty': True,
                'prev_day_used': 'N/A',
                'lookback_days': 0
            }
        
        # ✅ OPTIMIZED: Find most recent trading day using file index (no I/O)
        prev_day = day - pd.Timedelta(days=1)
        max_lookback = 7
        lookback_count = 0
        
        while lookback_count < max_lookback:
            prev_str = prev_day.strftime("%Y-%m-%d")
            # ✅ FAST: Check file index instead of loading data
            if prev_str in self._file_index and len(self._file_index[prev_str]) > 0:
                break
            lookback_count += 1
            prev_day = prev_day - pd.Timedelta(days=1)
        
        # If no previous trading day found within lookback window
        if lookback_count >= max_lookback:
            prev_str = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"No previous trading day found within {max_lookback} days for {today_str}")
            return {
                'gaps': pd.DataFrame(columns=['symbol', 'open_price', 'prev_close', 'last_price', 'gap_percent']),
                'prev_syms': set(),
                'today_syms': set(),
                'prev_empty': True,
                'today_empty': False,
                'prev_day_used': prev_str,
                'lookback_days': lookback_count
            }
        
        prev_str = prev_day.strftime("%Y-%m-%d")

        # ✅ NOW load data (only after confirming both days exist in index)
        today_df = self._get_day_df(today_str)
        prev_df = self._get_day_df(prev_str)

        out = {
            'gaps': pd.DataFrame(columns=['symbol', 'open_price', 'prev_close', 'last_price', 'gap_percent']),
            'prev_syms': set(),
            'today_syms': set(),
            'prev_empty': prev_df.empty,
            'today_empty': today_df.empty,
            'prev_day_used': prev_str,
            'lookback_days': lookback_count
        }
        
        if prev_df.empty or today_df.empty:
            return out

        col = 'symbol' if 'symbol' in today_df.columns else ('ticker' if 'ticker' in today_df.columns else None)
        if col is None:
            return out

        if premarket:
            day_naive = datetime(day.year, day.month, day.day, 4, 0)
        else:
            day_naive = datetime(day.year, day.month, day.day, 9, 30)
        
        prev_day_naive = datetime(prev_day.year, prev_day.month, prev_day.day, 16, 0)
        
        open_et = pd.Timestamp(day_naive, tz='US/Eastern')
        open_utc = open_et.tz_convert('UTC')

        close_et_prev = pd.Timestamp(prev_day_naive, tz='US/Eastern')
        close_utc_prev = close_et_prev.tz_convert('UTC')

        td = today_df[[col, 'timestamp', 'open', 'close']].dropna(subset=[col, 'timestamp']).copy()

        # Log timestamp info for debugging
        if not td.empty:
            sample_ts = td['timestamp'].iloc[0]
            logger.debug(f"Today's data: {len(td)} bars, sample timestamp: {sample_ts} (tz={sample_ts.tz}), "
                        f"open_utc threshold: {open_utc}, premarket: {premarket}")

        td = td[td['timestamp'] >= open_utc]
        if td.empty:
            out['today_empty'] = True
            logger.debug(f"No opening bars found for {today_str} after filtering for timestamp >= {open_utc}")
            return out
        td = td.sort_values('timestamp')
        first_idx = td.groupby(col)['timestamp'].head(1).index
        today_open = td.loc[first_idx][[col, 'open']].rename(columns={col: 'symbol', 'open': 'open_price'})

        pv = prev_df[[col, 'timestamp', 'close']].dropna(subset=[col, 'timestamp']).copy()

        # Log timestamp info for debugging
        if not pv.empty:
            sample_ts = pv['timestamp'].iloc[0]
            logger.debug(f"Previous day data: {len(pv)} bars, sample timestamp: {sample_ts} (tz={sample_ts.tz}), "
                        f"close_utc threshold: {close_utc_prev}")

        pv = pv[pv['timestamp'] <= close_utc_prev]
        if pv.empty:
            out['prev_empty'] = True
            logger.debug(f"No closing bars found for {prev_str} after filtering for timestamp <= {close_utc_prev}")
            return out
        pv = pv.sort_values('timestamp')
        last_idx = pv.groupby(col)['timestamp'].tail(1).index
        prev_close = pv.loc[last_idx][[col, 'close']].rename(columns={col: 'symbol', 'close': 'prev_close'})

        out['prev_syms'] = set(prev_close['symbol'].astype(str).unique())
        out['today_syms'] = set(today_open['symbol'].astype(str).unique())

        merged = pd.merge(today_open, prev_close, on='symbol', how='inner')
        if merged.empty:
            return out

        merged['last_price'] = merged['open_price']
        merged['gap_percent'] = ((merged['open_price'] - merged['prev_close']) / merged['prev_close']) * 100.0
        out['gaps'] = merged[['symbol', 'open_price', 'prev_close', 'last_price', 'gap_percent']].copy()
        
        # ✅ NEW: Log only if lookback was needed
        if lookback_count > 0:
            logger.debug(f"Gap calc for {today_str} used prev day {prev_str} ({lookback_count} days back)")
        
        return out

    def calculate_gaps_efficient(self, day: datetime, premarket: bool = False) -> Dict:
        """Optimized gap calculation - loads only required bars."""
        day = pd.Timestamp(day).normalize()
        prev_day = (day - pd.Timedelta(days=1)).normalize()
        
        today_str = day.strftime("%Y-%m-%d")
        prev_str = prev_day.strftime("%Y-%m-%d")
        
        cache_key = (today_str, premarket)
        if cache_key in self._gap_cache:
            self._gap_cache.move_to_end(cache_key)
            return self._gap_cache[cache_key]
        
        prev_exists = prev_str in self._file_index
        today_exists = today_str in self._file_index
        if not prev_exists or not today_exists:
            out = {'gaps': pd.DataFrame(), 'prev_empty': not prev_exists, 'today_empty': not today_exists}
            self._cache_result(cache_key, out)
            return out
        
        common_symbols = sorted(self._file_index[today_str] & self._file_index[prev_str])
        if not common_symbols:
            out = {'gaps': pd.DataFrame(), 'prev_empty': False, 'today_empty': False}
            self._cache_result(cache_key, out)
            return out
        
        if premarket:
            open_time = datetime(day.year, day.month, day.day, 4, 0)
        else:
            open_time = datetime(day.year, day.month, day.day, 9, 30)
        close_time = datetime(prev_day.year, prev_day.month, prev_day.day, 16, 0)
        
        open_et = pd.Timestamp(open_time, tz='US/Eastern')
        open_utc = open_et.tz_convert('UTC')
        close_et = pd.Timestamp(close_time, tz='US/Eastern')
        close_utc = close_et.tz_convert('UTC')
        
        gaps = []
        usable_prev_syms = set()
        usable_today_syms = set()
        skip_reasons = {}
        
        for symbol in common_symbols:
            gap_data = self._calculate_symbol_gap(
                symbol, today_str, prev_str, open_utc, close_utc, skip_reasons
            )
            if gap_data:
                gaps.append(gap_data)
                usable_today_syms.add(symbol)
                usable_prev_syms.add(symbol)
        
        result = {
            'gaps': pd.DataFrame(gaps),
            'prev_syms': usable_prev_syms,
            'today_syms': usable_today_syms,
            'prev_empty': False,
            'today_empty': False
        }
        
        self._log_gap_summary(today_str, premarket, len(common_symbols), len(gaps), skip_reasons)
        self._cache_result(cache_key, result)
        return result
    
    def _calculate_symbol_gap(self, symbol: str, today_str: str, prev_str: str,
                              open_utc, close_utc, skip_reasons: Dict) -> Dict:
        """Calculate gap for a single symbol."""
        today_df = self._get_symbol_day_data(symbol, today_str)
        if today_df.empty or 'timestamp' not in today_df.columns:
            skip_reasons.setdefault('no_today_file_or_timestamp', []).append(symbol)
            return None
        
        today_df = today_df.copy()
        today_df['timestamp'] = pd.to_datetime(today_df['timestamp'], utc=True, errors='coerce')
        today_bars = today_df[today_df['timestamp'] >= open_utc]
        
        if today_bars.empty:
            skip_reasons.setdefault('no_today_bars_after_open', []).append(symbol)
            return None
        
        today_open = today_bars.iloc[0].get('open')
        if pd.isna(today_open):
            skip_reasons.setdefault('today_open_nan', []).append(symbol)
            return None
        
        prev_df = self._get_symbol_day_data(symbol, prev_str)
        if prev_df.empty or 'timestamp' not in prev_df.columns:
            skip_reasons.setdefault('no_prev_file_or_timestamp', []).append(symbol)
            return None
        
        prev_df = prev_df.copy()
        prev_df['timestamp'] = pd.to_datetime(prev_df['timestamp'], utc=True, errors='coerce')
        prev_bars = prev_df[prev_df['timestamp'] <= close_utc]
        
        if prev_bars.empty:
            skip_reasons.setdefault('no_prev_bars_before_close', []).append(symbol)
            return None
        
        prev_close = prev_bars.iloc[-1].get('close')
        if pd.isna(prev_close) or prev_close == 0:
            skip_reasons.setdefault('prev_close_nan_or_zero', []).append(symbol)
            return None

        gap_pct = ((today_open - prev_close) / prev_close) * 100.0
        
        return {
            'symbol': symbol,
            'open_price': today_open,
            'prev_close': prev_close,
            'last_price': today_open,
            'gap_percent': gap_pct
        }
    
    def _cache_result(self, cache_key, result):
        """Cache a gap calculation result."""
        if len(self._gap_cache) >= self._gap_cache_limit:
            self._gap_cache.popitem(last=False)
        self._gap_cache[cache_key] = result
    
    def _log_gap_summary(self, today_str: str, premarket: bool, total: int, 
                        found: int, skip_reasons: Dict):
        """Log summary of gap calculation."""
        skipped = total - found
        summary_parts = []
        for reason, syms in skip_reasons.items():
            cnt = len(syms)
            examples = ",".join(sorted(syms)[:5])
            summary_parts.append(f"{reason}={cnt} [{examples}]")
        summary = "; ".join(summary_parts) if summary_parts else "none"
        logger.info(
            f"calculate_gaps_efficient {today_str} premarket={premarket} "
            f"total_symbols={total} found={found} skipped={skipped} skips=({summary})"
        )