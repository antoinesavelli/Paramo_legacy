# =====================================================
# news.backtest.py - Backtest News Integration (Parquet)
# =====================================================

"""News data integration for backtesting with time-aware filtering."""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import pandas as pd
from utils.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)

class NewsIntegrationBacktest:
    """
    Time-aware news integration: Filter by negative sentiment and enforce temporal constraints.
    
    Rules:
    1. If any news has neg > 0.08, reject symbol entirely
    2. If all news has neg <= 0.08, allow trading only AFTER the earliest news timestamp
    3. Hold only one monthly file in memory at a time
    """

    def __init__(self, config, data_dir: str):
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = get_logger(__name__, component="news_bt")
        
        # ✅ Single monthly cache (not 30)
        self._current_month_key = None
        self._current_month_data = None

    def _load_month_data(self, year: int, month: int) -> pd.DataFrame:
        """Load monthly news file with caching (single file only)."""
        month_key = f"{year}{month:02d}"
        
        # Return cached data if already loaded
        if self._current_month_key == month_key and self._current_month_data is not None:
            return self._current_month_data
        
        # Load new monthly file
        file_path = self.data_dir / str(year) / f"news_{year}{month:02d}.parquet"
        
        if not file_path.exists():
            self.logger.debug(f"No news file: {file_path}")
            return pd.DataFrame()
        
        try:
            monthly_df = pd.read_parquet(file_path)
            
            # ✅ Parse date column as timezone-aware datetime (UTC)
            if 'date' in monthly_df.columns:
                monthly_df['date'] = pd.to_datetime(monthly_df['date'], utc=True)
            
            # Cache this month's data
            self._current_month_key = month_key
            self._current_month_data = monthly_df
            
            self.logger.debug(f"Loaded news: {len(monthly_df)} articles for {year}-{month:02d}")
            return monthly_df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def check_news_approval(
        self, 
        symbol: str, 
        current_time: pd.Timestamp
    ) -> Dict:
        """
        Check if trading is approved for a symbol at a specific time.
        
        Args:
            symbol: Stock symbol
            current_time: Current timestamp (UTC timezone-aware)
            
        Returns:
            {
                'approved': bool,              # True if trading allowed
                'reason': str,                 # Rejection reason if not approved
                'earliest_news_time': pd.Timestamp or None,  # Earliest news timestamp
                'article_count': int,          # Total articles found
                'max_negative': float,         # Highest negative sentiment
                'articles_before_time': int    # Articles published before current_time
            }
        """
        # Ensure current_time is timezone-aware (UTC)
        if current_time.tz is None:
            current_time = current_time.tz_localize('UTC')
        else:
            current_time = current_time.tz_convert('UTC')
        
        # Load appropriate monthly file
        year = current_time.year
        month = current_time.month
        monthly_df = self._load_month_data(year, month)
        
        if monthly_df.empty:
            return {
                'approved': False,
                'reason': 'no_news_data',
                'earliest_news_time': None,
                'article_count': 0,
                'max_negative': 0.0,
                'articles_before_time': 0
            }
       
