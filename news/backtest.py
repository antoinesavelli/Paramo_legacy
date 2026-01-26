# =====================================================
# news.backtest.py - Backtest News Integration (Parquet)
# =====================================================

"""News data integration for backtesting."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import pandas as pd
from utils.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)

class NewsIntegrationBacktest:
    """
    Simplified news integration: Check for presence of news and filter negative sentiment.
    """

    def __init__(self, config, data_dir: str):
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = get_logger(__name__, component="news_bt")
        # Cache to avoid re-reading monthly files
        self._news_cache = {}
        self._news_cache_limit = 30  # Store about a month's worth

    def get_news_for_date(self, date: datetime, symbol: str = None) -> pd.DataFrame:
        """
        Get news for specific date (optionally filtered by symbol).
        
        Returns DataFrame with columns:
        - date, symbol, title, link, polarity, neg, neu, pos
        """
        year = date.year
        month = f"{date.month:02d}"
        month_key = f"{year}{month}"
        
        # Check cache
        if month_key in self._news_cache:
            monthly_df = self._news_cache[month_key]
        else:
            # Load monthly file: T:\news_data\YYYY\news_YYYYMM.parquet
            file_path = self.data_dir / str(year) / f"news_{year}{month}.parquet"
            
            if not file_path.exists():
                self.logger.debug(f"No news file: {file_path}")
                return pd.DataFrame()
            
            try:
                monthly_df = pd.read_parquet(file_path)
                
                # Cache management
                if len(self._news_cache) >= self._news_cache_limit:
                    oldest = list(self._news_cache.keys())[0]
                    del self._news_cache[oldest]
                
                self._news_cache[month_key] = monthly_df
                self.logger.debug(f"Loaded news: {len(monthly_df)} articles for {year}-{month}")
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                return pd.DataFrame()
        
        if monthly_df.empty:
            return pd.DataFrame()
        
        # Filter to specific date
        target_date = pd.Timestamp(date).normalize()
        daily_news = monthly_df[monthly_df['date'].dt.normalize() == target_date].copy()
        
        # Filter to symbol if provided
        if symbol and not daily_news.empty:
            daily_news = daily_news[daily_news['symbol'].str.upper() == str(symbol).upper()]
        
        return daily_news

    def analyze_news_impact(self, symbol: str, date: datetime) -> Dict:
        """
        Simplified analysis: Check for news presence and negative sentiment.
        
        Returns:
            {
                'has_news': bool,           # Any news articles found
                'article_count': int,       # Number of articles
                'avg_negative': float,      # Average negative sentiment (0-1)
                'max_negative': float,      # Highest negative sentiment (0-1)
                'passes_filter': bool       # True if passes negative sentiment filter
            }
        """
        news_df = self.get_news_for_date(date, symbol)
        
        if news_df.empty:
            return {
                'has_news': False,
                'article_count': 0,
                'avg_negative': 0.0,
                'max_negative': 0.0,
                'passes_filter': False  # No news = fail
            }
        
        article_count = len(news_df)
        
        # Check negative sentiment column
        if 'neg' not in news_df.columns:
            self.logger.warning(f"{symbol}: 'neg' column missing, assuming 0.0")
            avg_negative = 0.0
            max_negative = 0.0
        else:
            avg_negative = float(news_df['neg'].mean())
            max_negative = float(news_df['neg'].max())
        
        # Get threshold from config
        max_neg_threshold = getattr(
            self.config.backtest, 
            'MAX_NEGATIVE_SENTIMENT', 
            0.05
        )
        
        # Pass if has news AND negative sentiment is acceptable
        passes_filter = (article_count > 0) and (max_negative <= max_neg_threshold)
        
        return {
            'has_news': True,
            'article_count': article_count,
            'avg_negative': avg_negative,
            'max_negative': max_negative,
            'passes_filter': passes_filter
        }
