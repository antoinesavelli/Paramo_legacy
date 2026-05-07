# =====================================================
# File index cache system for fast date-range specific data loading.
# =====================================================

"""
File index cache system for fast date-range specific data loading.

This module creates and manages a pre-built index of available data files
for specific date ranges, eliminating the need to scan directories on every backtest run.

Performance improvements:
- Pre-scan date range once, cache results
- Instant loading on subsequent runs (date range match only)
- Pickle-based serialization for fast I/O
- Dedicated cache storage in trading_data directory

FILE FORMAT: YYYY-MM-DD.parquet (ISO date format)
"""

import logging
from utils.logging import get_logger
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List
from datetime import datetime, timedelta

logger = get_logger(__name__)


class FileIndexCache:
    """
    Manages cached file indices for date ranges.
    
    Cache structure:
    {
        'date_range': (start_date, end_date),
        'file_index': {
            'YYYY-MM-DD': {symbol1, symbol2, ...},
            ...
        },
        'aggregate_index': {
            'YYYYMM': True/False,
            ...
        },
        'created_at': timestamp,
        'data_dir': str,
        'aggregate_dir': str,
        'total_days': int,
        'total_symbols': int
    }
    
    Cache file naming: index_STARTDATE_ENDDATE.pkl
    Example: index_20240103_20241231.pkl
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: Directory to store cache files.
                       Default is a portable relative path; override via
                       config.system.FILE_INDEX_CACHE_DIR or pass explicitly.
        """
        if cache_dir is None:
            cache_dir = "data/cache/file_index"
        
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug("FileIndexCache initialized: %s", self._cache_dir)
    
    def get_or_build_index(self, 
                          data_dir: str,
                          aggregate_dir: str,
                          start_date: str,
                          end_date: str,
                          force_rebuild: bool = False) -> Dict:
        """
        Get cached index or build new one for date range.
        
        Validation: Only checks if date range matches (simple and fast).
        
        Args:
            data_dir: Intraday data directory (S:\\trading\\ticker_data)
            aggregate_dir: Daily aggregates directory (S:\\trading\\ticker_data\\daily_aggregates)
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            force_rebuild: Force rebuild even if cache exists
            
        Returns:
            Dict with:
                - 'file_index': {date: Set[symbols]}
                - 'aggregate_index': {month: bool}
                - 'date_range': (start, end)
                - 'cached': bool (was this from cache?)
                - 'total_days': int
                - 'total_symbols': int
        """
        # Generate cache file name from dates
        cache_file = self._get_cache_file_path(start_date, end_date)
        
        # Try to load from cache
        if not force_rebuild and cache_file.exists():
            cached_data = self._load_cache(cache_file)
            
            if cached_data and self._validate_cache(cached_data, start_date, end_date):
                logger.info("✓ Loaded file index from cache: %s", cache_file.name)
                cached_data['cached'] = True
                return cached_data
            else:
                logger.warning("Cache invalid - rebuilding")
        
        # Build new index
        logger.info("Building file index for %s to %s...", start_date, end_date)
        
        index_data = self._build_index(
            data_dir=data_dir,
            aggregate_dir=aggregate_dir,
            start_date=start_date,
            end_date=end_date
        )
        
        # Save to cache
        self._save_cache(cache_file, index_data)
        
        logger.info("✓ File index built and cached: %s", cache_file.name)
        index_data['cached'] = False
        
        return index_data
    
    def _build_index(self, data_dir: str, aggregate_dir: str,
                     start_date: str, end_date: str) -> Dict:
        """
        Build file index for date range.
        
        Returns:
            Dict with file_index and aggregate_index
        """
        data_path = Path(data_dir)
        agg_path = Path(aggregate_dir)
        
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        # Build intraday file index
        file_index = {}
        current_date = start_dt
        
        logger.info("Scanning intraday files from %s to %s...", start_date, end_date)
        
        total_files = (end_dt - start_dt).days + 1
        scanned = 0
        
        while current_date <= end_dt:
            date_str = current_date.strftime("%Y-%m-%d")
            year = current_date.strftime("%Y")
            month = current_date.strftime("%m")
            
            # NOTE: NEW FORMAT: YYYY-MM-DD.parquet
            file_path = data_path / year / month / f"{date_str}.parquet"
            
            if file_path.exists():
                # Quick scan for symbols
                try:
                    # NOTE: CHANGED: Read symbol column directly (most efficient)
                    df_symbols = pd.read_parquet(file_path, columns=['symbol'])
                    symbols = set(df_symbols['symbol'].dropna().unique())
                    symbols = {s for s in symbols if s and isinstance(s, str)}
                    
                    file_index[date_str] = symbols
                    scanned += 1
                    
                    if scanned % 10 == 0:
                        logger.info("  Progress: %d/%d files scanned", scanned, total_files)
                    
                except Exception as e:
                    logger.error("Error scanning %s: %s", file_path, e, exc_info=True)
                    file_index[date_str] = set()
            else:
                # Don't add empty sets for missing files
                pass
            
            current_date += pd.Timedelta(days=1)
        
        logger.info("✓ Scanned %d intraday files", scanned)
        
        # Build aggregate file index
        aggregate_index = {}
        months_needed = self._get_months_in_range(start_dt, end_dt)
        
        logger.info("Scanning aggregate files for %d months...", len(months_needed))
        
        for year_month in months_needed:
            year = year_month[:4]
            month_file = agg_path / year / f"{year_month}.parquet"
            aggregate_index[year_month] = month_file.exists()
            
            status = "✓" if month_file.exists() else "✗"
            logger.debug("  %s: %s", year_month, status)
        
        logger.info("✓ Scanned %d aggregate files", len(months_needed))
        
        # Calculate statistics
        days_with_data = len([d for d in file_index.values() if d])
        all_symbols = set.union(*file_index.values()) if file_index else set()
        
        return {
            'date_range': (start_date, end_date),
            'file_index': file_index,
            'aggregate_index': aggregate_index,
            'created_at': datetime.now().isoformat(),
            'data_dir': str(data_path),
            'aggregate_dir': str(agg_path),
            'total_days': days_with_data,
            'total_symbols': len(all_symbols)
        }
    
    def _get_months_in_range(self, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> List[str]:
        """Get list of YYYYMM strings covering date range."""
        months = set()
        current = start_dt
        
        # Include month before start (for prev_close lookups)
        prev_month = start_dt - pd.DateOffset(months=1)
        months.add(prev_month.strftime("%Y%m"))
        
        while current <= end_dt:
            months.add(current.strftime("%Y%m"))
            current += pd.DateOffset(months=1)
        
        return sorted(months)
    
    def _get_cache_file_path(self, start_date: str, end_date: str) -> Path:
        """Generate cache file path from date range."""
        start_compact = start_date.replace('-', '')
        end_compact = end_date.replace('-', '')
        filename = f"index_{start_compact}_{end_compact}.pkl"
        return self._cache_dir / filename
    
    def _validate_cache(self, cached_data: Dict, start_date: str, end_date: str) -> bool:
        """Validate cached data matches requested date range."""
        if not cached_data or 'date_range' not in cached_data:
            return False
        
        cached_start, cached_end = cached_data['date_range']
        return cached_start == start_date and cached_end == end_date
    
    def _load_cache(self, cache_file: Path) -> Optional[Dict]:
        """Load cached index from pickle file."""
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.error("Error loading cache %s: %s", cache_file, e, exc_info=True)
            return None
    
    def _save_cache(self, cache_file: Path, data: Dict):
        """Save index data to pickle file."""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Saved cache: %s", cache_file)
        except Exception as e:
            logger.error("Error saving cache %s: %s", cache_file, e, exc_info=True)


def create_index_for_backtest(config) -> Dict:
    """
    Convenience function to create/load index for backtest config.
    
    Args:
        config: TradingConfig object with backtest settings
        
    Returns:
        Index data dict
    """
    cache = FileIndexCache(
        cache_dir=config.system.FILE_INDEX_CACHE_DIR
        if hasattr(config, 'system') and hasattr(config.system, 'FILE_INDEX_CACHE_DIR')
        else None
    )
    
    data_dir = config.backtest.DATA_DIR
    aggregate_dir = Path(data_dir) / "daily_aggregates"
    start_date = config.backtest.START_DATE
    end_date = config.backtest.END_DATE
    
    return cache.get_or_build_index(
        data_dir=data_dir,
        aggregate_dir=str(aggregate_dir),
        start_date=start_date,
        end_date=end_date
    )