# =====================================================
# news.backtest.py - Backtest News Integration (Parquet)
# =====================================================

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import pandas as pd
from utils.logging import get_logger

try:
    import pyarrow.parquet as pq  # optional, for metadata checks
except ImportError:  # pragma: no cover
    pq = None

class NewsIntegrationBacktest:
    """
    Backtest mode: per-day Parquet at NEWS_DATA_DIR/YYYY-MM-DD.parquet
    Schema supported (aggregated one row per (date, symbol)):
      - date (date32), symbol (utf8), article_count (int64),
        avg_sentiment (double in [-1,1]), avg_match_score (double in [0,100]),
        total_highlights (int64), source_domains (list<utf8>), top_headlines (list<utf8>),
        created_at (timestamp[ns], naive), article_uuids (list<utf8>)
    """

    def __init__(self, config, data_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.logger = get_logger(__name__, component="news_bt")
        self.catalyst_keywords = {
            'fda': ['FDA', 'approval', 'clearance', 'PDUFA', 'clinical trial', 'Phase 3'],
            'earnings': ['earnings', 'revenue', 'beat', 'guidance', 'EPS'],
            'merger': ['merger', 'acquisition', 'buyout', 'takeover', 'deal'],
            'partnership': ['partnership', 'collaboration', 'agreement', 'contract'],
            'product': ['launch', 'release', 'announcement', 'breakthrough'],
            'regulatory': ['SEC', 'investigation', 'compliance', 'regulatory']
        }
        # Add a news cache to avoid re-reading files
        self._news_cache = {}
        self._news_cache_limit = 30  # Store about a month's worth of data

    def analyze_news_impact(self, symbol: str, date=None) -> Dict:
        if not self.data_dir or date is None:
            return self._empty_analysis()
        try:
            day_dt = pd.to_datetime(date).normalize()
        except Exception:
            return self._empty_analysis()

        day_str = day_dt.strftime("%Y-%m-%d")
        
        # Check cache first
        cache_key = day_str
        if cache_key in self._news_cache:
            df = self._news_cache[cache_key]
        else:
            file_path = os.path.join(self.data_dir, f"{day_str}.parquet")
            if not os.path.exists(file_path):
                return self._empty_analysis()

            try:
                df = pd.read_parquet(file_path)
                # Cache the result
                if len(self._news_cache) >= self._news_cache_limit:
                    # Remove oldest item (simple implementation)
                    oldest = list(self._news_cache.keys())[0]
                    del self._news_cache[oldest]
                self._news_cache[cache_key] = df
            except Exception as e:
                self.logger.error(f"Error reading day news file {file_path}: {e}")
                return self._empty_analysis()
                
        if df is None or df.empty:
            return self._empty_analysis()

        # Standardize symbol column and filter
        sym_col = next((c for c in ('symbol', 'ticker', 'Symbol', 'Ticker') if c in df.columns), None)
        if not sym_col:
            return self._empty_analysis()
        try:
            df[sym_col] = df[sym_col].astype(str).str.upper()
            df = df[df[sym_col] == str(symbol).upper()]
        except Exception:
            pass
        if df.empty:
            return self._empty_analysis()

        # Prefer 'date' col if present
        if 'date' in df.columns:
            try:
                dcol = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
                df = df[dcol == day_dt]
            except Exception:
                pass
        if df.empty:
            return self._empty_analysis()

        # Choose latest created_at if present
        if 'created_at' in df.columns:
            try:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')  # naive local
                df = df.sort_values('created_at').tail(1)
            except Exception:
                df = df.tail(1)
        else:
            df = df.tail(1)

        row = df.iloc[0]

        # Catalyst type
        catalyst_type = None
        if 'catalyst_type' in df.columns and pd.notna(row.get('catalyst_type')):
            catalyst_type = str(row.get('catalyst_type'))
        else:
            headlines = []
            if 'top_headlines' in df.columns and pd.notna(row.get('top_headlines')):
                try:
                    headlines = list(row.get('top_headlines') or [])
                except Exception:
                    headlines = []
            joined = " ".join([h for h in headlines if isinstance(h, str)])
            catalyst_type = self._identify_catalyst(joined) if joined else 'general'

        # Strength from aggregated metrics
        article_count = int(row.get('article_count')) if 'article_count' in df.columns and pd.notna(row.get('article_count')) else max(1, len(df))
        avg_sent = float(row.get('avg_sentiment')) if 'avg_sentiment' in df.columns and pd.notna(row.get('avg_sentiment')) else 0.0  # [-1,1]
        match_score = float(row.get('avg_match_score')) if 'avg_match_score' in df.columns and pd.notna(row.get('avg_match_score')) else 50.0  # [0,100]
        highlights = int(row.get('total_highlights')) if 'total_highlights' in df.columns and pd.notna(row.get('total_highlights')) else 0

        count_score = min(40.0, 5.0 * max(0, article_count))         # 8+ -> 40
        match_score_s = 0.40 * match_score                            # 0..40
        sent_score = 15.0 * ((avg_sent + 1.0) / 2.0)                  # [-1..1] -> 0..15
        hl_score = min(5.0, 0.5 * max(0, highlights))                 # 0..5
        strength = int(min(100.0, count_score + match_score_s + sent_score + hl_score))

        recent_count = 0
        if 'created_at' in df.columns and pd.notna(row.get('created_at')):
            try:
                created_at = pd.to_datetime(row.get('created_at'), errors='coerce')
                if pd.notna(created_at):
                    recent_count = 1 if (datetime.now() - created_at) < timedelta(hours=1) else 0
            except Exception:
                recent_count = 0

        payload = {
            'article_count': article_count,
            'avg_sentiment': avg_sent,
            'avg_match_score': match_score,
            'total_highlights': highlights
        }
        if 'top_headlines' in df.columns and pd.notna(row.get('top_headlines')):
            try:
                payload['top_headlines'] = list(row.get('top_headlines') or [])[:3]
            except Exception:
                payload['top_headlines'] = []
        if 'source_domains' in df.columns and pd.notna(row.get('source_domains')):
            try:
                payload['source_domains'] = list(row.get('source_domains') or [])[:10]
            except Exception:
                payload['source_domains'] = []

        return {
            'has_catalyst': True,
            'catalyst_strength': strength,
            'catalyst_type': catalyst_type,
            'news_count': int(article_count),
            'recent_count': int(recent_count),
            'news_items': [payload]
        }

    def _identify_catalyst(self, text: str) -> str:
        lower = text.lower()
        for ctype, words in self.catalyst_keywords.items():
            for w in words:
                if w.lower() in lower:
                    return ctype
        return 'general'

    @staticmethod
    def _empty_analysis() -> Dict:
        return {
            'has_catalyst': False,
            'catalyst_strength': 0,
            'catalyst_type': None,
            'news_count': 0,
            'recent_count': 0,
            'news_items': []
        }
