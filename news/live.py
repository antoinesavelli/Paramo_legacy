# =====================================================
# news.live.py - Live News and Catalyst Integration
# =====================================================

import requests
from bs4 import BeautifulSoup
from typing import Dict, List
from datetime import datetime, timedelta, timezone
import os
from utils import get_logger

class NewsIntegrationLive:
    """Live mode: fetch news from APIs/sites and assess catalyst strength."""

    NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
    ALPHAV_ENDPOINT = "https://www.alphavantage.co/query"
    REQUEST_TIMEOUT = 5
    SCRAPE_HEADERS = {'User-Agent': 'Mozilla/5.0'}

    def __init__(self, config):
        self.config = config
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', '')
        self.catalyst_keywords = {
            'fda': ['FDA', 'approval', 'clearance', 'PDUFA', 'clinical trial', 'Phase 3'],
            'earnings': ['earnings', 'revenue', 'beat', 'guidance', 'EPS'],
            'merger': ['merger', 'acquisition', 'buyout', 'takeover', 'deal'],
            'partnership': ['partnership', 'collaboration', 'agreement', 'contract'],
            'product': ['launch', 'release', 'announcement', 'breakthrough'],
            'regulatory': ['SEC', 'investigation', 'compliance', 'regulatory']
        }
        self._news_cache: Dict[str, List[Dict]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self.logger = get_logger(__name__, component="news_live")

    def analyze_news_impact(self, symbol: str, date=None) -> Dict:
        """Aggregate recent live news and compute a catalyst score."""
        news_items = self.get_news_for_symbol(symbol)
        if not news_items:
            return self._empty_analysis()

        catalyst_counts: Dict[str, int] = {}
        for item in news_items:
            ctype = item.get('catalyst_type', 'general')
            catalyst_counts[ctype] = catalyst_counts.get(ctype, 0) + 1

        primary_catalyst = max(catalyst_counts, key=catalyst_counts.get)
        strength = min(100, len(news_items) * 10)
        if primary_catalyst in ['fda', 'merger', 'earnings']:
            strength = min(100, int(strength * 1.5))

        now_utc = datetime.now(timezone.utc)
        recent_news = [n for n in news_items if (now_utc - n['timestamp']).total_seconds() < 3600]
        if recent_news:
            strength = min(100, int(strength * 1.2))

        result = {
            'has_catalyst': True,
            'catalyst_strength': strength,
            'catalyst_type': primary_catalyst,
            'news_count': len(news_items),
            'recent_count': len(recent_news),
            'news_items': news_items[:5]
        }
        self.logger.info(f"News analyzed {symbol}: has_catalyst={result['has_catalyst']} strength={result['catalyst_strength']} type={result['catalyst_type']}")
        return result

    def get_news_for_symbol(self, symbol: str) -> List[Dict]:
        """Fetch live news; cached briefly."""
        if self._is_cache_valid(symbol):
            return self._news_cache[symbol]

        items: List[Dict] = []
        items.extend(self._fetch_newsapi(symbol))
        items.extend(self._fetch_alpha_vantage_news(symbol))
        items.extend(self._scrape_financial_sites(symbol))

        # Deduplicate by (title,url)
        seen = set()
        deduped = []
        for n in items:
            key = (n.get('title'), n.get('url'))
            if key not in seen:
                seen.add(key)
                deduped.append(n)

        # Sort newest first
        deduped.sort(key=lambda x: x['timestamp'], reverse=True)

        self._news_cache[symbol] = deduped
        self._cache_expiry[symbol] = datetime.now(timezone.utc) + timedelta(minutes=15)
        self.logger.info(f"Fetched news for {symbol}: {len(deduped)} items")
        return deduped

    def _fetch_newsapi(self, symbol: str) -> List[Dict]:
        if not self.news_api_key:
            return []
        try:
            params = {
                'q': symbol,
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.utcnow() - timedelta(days=1)).isoformat(timespec='seconds') + "Z",
                'pageSize': 50
            }
            resp = requests.get(self.NEWSAPI_ENDPOINT, params=params, timeout=self.REQUEST_TIMEOUT)
            if resp.status_code != 200:
                self.logger.debug(f"NewsAPI non-200 {resp.status_code}: {resp.text[:200]}")
                return []
            data = resp.json()
            articles = data.get('articles', []) or []
            out = []
            for a in articles:
                published = a.get('publishedAt')
                try:
                    ts = datetime.fromisoformat(published.replace('Z', '+00:00'))
                except Exception:
                    ts = datetime.now(timezone.utc)
                out.append({
                    'source': (a.get('source') or {}).get('name', ''),
                    'title': a.get('title', ''),
                    'description': a.get('description', '') or '',
                    'url': a.get('url', ''),
                    'timestamp': ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
                    'catalyst_type': self._identify_catalyst(f"{a.get('title','')} {a.get('description','')}")
                })
            return out
        except Exception as e:
            self.logger.debug(f"NewsAPI error for {symbol}: {e}")
            return []

    def _fetch_alpha_vantage_news(self, symbol: str) -> List[Dict]:
        if not self.alpha_vantage_key:
            return []
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_key
            }
            resp = requests.get(self.ALPHAV_ENDPOINT, params=params, timeout=self.REQUEST_TIMEOUT)
            if resp.status_code != 200:
                self.logger.debug(f"AlphaVantage non-200 {resp.status_code}: {resp.text[:200]}")
                return []
            data = resp.json()
            if any(k in data for k in ('Note', 'Information', 'Error Message')):
                self.logger.debug(f"AlphaVantage message for {symbol}: {data}")
                return []
            feed = data.get('feed', []) or []
            results = []
            for item in feed[:10]:
                raw_ts = item.get('time_published', '')
                try:
                    ts = datetime.strptime(raw_ts, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
                except Exception:
                    ts = datetime.now(timezone.utc)
                title = item.get('title', '')
                summary = item.get('summary', '')
                results.append({
                    'source': item.get('source', ''),
                    'title': title,
                    'description': summary,
                    'url': item.get('url', ''),
                    'timestamp': ts,
                    'sentiment': item.get('overall_sentiment_score', 0),
                    'catalyst_type': self._identify_catalyst(f"{title} {summary}")
                })
            return results
        except Exception as e:
            self.logger.debug(f"AlphaVantage error for {symbol}: {e}")
            return []

    def _scrape_financial_sites(self, symbol: str) -> List[Dict]:
        items: List[Dict] = []
        sites = [
            {
                'url': f'https://finance.yahoo.com/quote/{symbol}/news',
                'selector': 'h3 a',
                'source': 'Yahoo Finance',
                'base': 'https://finance.yahoo.com'
            },
            {
                'url': f'https://www.benzinga.com/quote/{symbol}/news',
                'selector': '.news-title',
                'source': 'Benzinga',
                'base': 'https://www.benzinga.com'
            }
        ]
        for site in sites:
            try:
                resp = requests.get(site['url'], headers=self.SCRAPE_HEADERS, timeout=self.REQUEST_TIMEOUT)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.content, 'html.parser')
                for h in soup.select(site['selector'])[:5]:
                    title = h.get_text(strip=True)
                    href = h.get('href', '')
                    if href and href.startswith('/'):
                        href = site['base'] + href
                    items.append({
                        'source': site['source'],
                        'title': title,
                        'description': '',
                        'url': href,
                        'timestamp': datetime.now(timezone.utc),
                        'catalyst_type': self._identify_catalyst(title)
                    })
            except Exception as e:
                self.logger.debug(f"Scrape error {site['source']} {symbol}: {e}")
        return items

    def _identify_catalyst(self, text: str) -> str:
        lower = text.lower()
        for ctype, words in self.catalyst_keywords.items():
            for w in words:
                if w.lower() in lower:
                    return ctype
        return 'general'

    def _is_cache_valid(self, symbol: str) -> bool:
        exp = self._cache_expiry.get(symbol)
        return bool(exp and datetime.now(timezone.utc) < exp)

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
