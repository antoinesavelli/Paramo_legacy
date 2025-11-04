# =====================================================
# data_handler.api.py - Market Data Management
# =====================================================

import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime
import pytz
from typing import Dict, List, Optional
import time
from config import TradingConfig
from utils.logging import get_logger
from utils.helpers import log_and_return
import yfinance as yf

def log_data_error(logger, msg, exc):
    logger.error(f"{msg}: {exc}")

def fetch_bars(api, symbol, timeframe, limit, logger):
    """Helper to fetch bars with error handling."""
    try:
        bars = api.get_bars(symbol, timeframe, limit=limit, end=datetime.now()).df
        return bars
    except Exception as e:
        log_data_error(logger, f"Error fetching bars for {symbol}", e)
        return pd.DataFrame()

class APIDataHandler:
    """Handles all market data retrieval and processing (live/API)"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.api = tradeapi.REST(
            config.api.ALPACA_API_KEY,
            config.api.ALPACA_SECRET_KEY,
            config.api.ALPACA_BASE_URL,
            api_version='v2'
        )
        self.logger = get_logger(__name__, component="data_handler")
        self.eastern = pytz.timezone('US/Eastern')
        self._cache = {}
        self._last_update = {}

    def get_market_status(self) -> Dict:
        """Check if market is open and trading status"""
        try:
            clock = self.api.get_clock()
            status = {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close,
                'timestamp': datetime.now(pytz.UTC).astimezone(self.eastern)
            }
            return status
        except Exception as e:
            return log_and_return(self.logger, f"Error getting market status: {e}", {'is_open': False})

    def get_universe(self) -> pd.DataFrame:
        """Get list of all tradeable stocks with basic filters"""
        try:
            assets = self.api.list_assets(status='active')
            df = pd.DataFrame([{
                'symbol': asset.symbol,
                'name': asset.name,
                'exchange': asset.exchange,
                'tradable': asset.tradable,
                'shortable': asset.shortable,
                'marginable': asset.marginable,
                'easy_to_borrow': asset.easy_to_borrow
            } for asset in assets])
            df = df[(df['tradable'] == True) & (df['exchange'].isin(['NYSE', 'NASDAQ', 'ARCA', 'AMEX']))]
            self.logger.info(f"Universe built: {len(df)} symbols")
            return df
        except Exception as e:
            log_data_error(self.logger, "Error getting universe", e)
            return pd.DataFrame()

    def get_quote_data(self, symbols: List[str]) -> Dict:
        """Get real-time quote data for multiple symbols"""
        quotes = {}
        batch_size = self.config.system.BATCH_SIZE
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            try:
                batch_quotes = self.api.get_quotes(batch)
                for symbol, quote in batch_quotes.items():
                    if quote:
                        quotes[symbol] = {
                            'bid': quote.bid_price,
                            'ask': quote.ask_price,
                            'bid_size': quote.bid_size,
                            'ask_size': quote.ask_size,
                            'last': quote.last,
                            'volume': quote.volume,
                            'timestamp': quote.timestamp
                        }
                self.logger.debug(f"Fetched quotes batch size={len(batch)} received={len(batch_quotes)}")
            except Exception as e:
                log_data_error(self.logger, "Error getting quotes for batch", e)
        self.logger.info(f"Quotes fetched for {len(quotes)} symbols (requested={len(symbols)})")
        return quotes

    def calculate_gaps(self, symbols: List[str]) -> pd.DataFrame:
        """Calculate gap percentages for screening"""
        gaps = []
        skipped = 0
        for symbol in symbols:
            bars = fetch_bars(self.api, symbol, '1Day', 2, self.logger)
            if len(bars) >= 2:
                prev_close = bars.iloc[-2]['close']
                current_price = bars.iloc[-1]['open']
                gap_percent = ((current_price - prev_close) / prev_close) * 100
                gaps.append({
                    'symbol': symbol,
                    'prev_close': prev_close,
                    'current_price': current_price,
                    'gap_percent': gap_percent
                })
            else:
                skipped += 1
        self.logger.info(f"Gap calc complete: computed={len(gaps)} skipped={skipped}")
        return pd.DataFrame(gaps)

    def get_intraday_bars(self, symbol: str, timeframe: str = '1Min', limit: int = 390) -> pd.DataFrame:
        """Get intraday bar data for pattern analysis"""
        max_retries = self.config.system.MAX_RETRIES
        for attempt in range(max_retries):
            bars = fetch_bars(self.api, symbol, timeframe, limit, self.logger)
            if not bars.empty:
                bars['symbol'] = symbol
                bars['vwap'] = (bars['close'] * bars['volume']).cumsum() / bars['volume'].cumsum()
                bars['dollar_volume'] = bars['close'] * bars['volume']
                if attempt > 0:
                    self.logger.info(f"Bars fetched for {symbol} on retry {attempt+1}: {len(bars)} rows")
                return bars
            self.logger.warning(f"Attempt {attempt+1} failed for {symbol}: empty bars")
            time.sleep(2)
        self.logger.error(f"All retries failed. Could not retrieve data for {symbol}.")
        return pd.DataFrame()

    def get_float_data(self, symbol: str) -> Optional[Dict]:
        """Get float and shares outstanding data"""
        try:
            fundamentals = self.api.get_fundamentals(symbol)
            if fundamentals:
                data = {
                    'shares_outstanding': fundamentals.shares_outstanding,
                    'float': fundamentals.float_shares,
                    'market_cap': fundamentals.market_cap
                }
                self.logger.debug(f"Alpaca fundamentals for {symbol} received")
                return data
        except Exception as e:
            log_data_error(self.logger, f"Error getting Alpaca fundamentals for {symbol}", e)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            data = {
                'shares_outstanding': info.get('sharesOutstanding'),
                'float': info.get('floatShares'),
                'market_cap': info.get('marketCap')
            }
            self.logger.debug(f"yfinance fundamentals for {symbol} received")
            return data
        except Exception as e:
            log_data_error(self.logger, f"Error getting yfinance fundamentals for {symbol}", e)
            return None