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
from data_handler.gap.gap_calculator import GapCalculator

def fetch_bars(api, symbol, timeframe, limit, logger):
    """Helper to fetch bars with error handling."""
    try:
        bars = api.get_bars(symbol, timeframe, limit=limit, end=datetime.now()).df
        return bars
    except Exception as e:
        logger.error("Error fetching bars for %s: %s", symbol, e)
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
        
        # NOTE: Initialize GapCalculator for live mode (deferred - universe loaded later)
        self.gap_calculator = None
    
    def get_universe(self) -> pd.DataFrame:
        """Get list of tradeable stocks from symbols.csv (required)"""
        from pathlib import Path
        
        # Try loading symbols.csv (market-cap filtered universe)
        # Check multiple possible locations
        possible_paths = [
            Path(self.config.backtest.DATA_DIR) / "symbols.csv",
            Path("data_handler") / "symbols.csv",
            Path("symbols.csv")
        ]
        
        for symbols_csv in possible_paths:
            if symbols_csv.exists():
                try:
                    df = pd.read_csv(symbols_csv)
                    # Normalize column name
                    if 'Symbol' in df.columns:
                        df = df.rename(columns={'Symbol': 'symbol'})
                    
                    if 'symbol' in df.columns:
                        df = df[['symbol']].dropna().drop_duplicates().reset_index(drop=True)
                        df['symbol'] = df['symbol'].str.upper()
                        self.logger.info(f"Loaded universe from {symbols_csv}: {len(df)} symbols")
                        
                        # NOTE: After loading universe, initialize gap calculator
                        if self.gap_calculator is None and not df.empty:
                            universe_symbols = set(df['symbol'].unique())
                            self.gap_calculator = GapCalculator(
                                get_daily_stats_func=self._get_daily_stats_from_api,
                                file_index=None,  # No file index for live
                                universe_symbols=universe_symbols  # NOTE: Pass universe
                            )
                            self.logger.info(f"Initialized gap calculator with {len(universe_symbols)} symbols")
                        
                        return df
                except Exception as e:
                    self.logger.warning(f"Failed to load {symbols_csv}: {e}")
        
        # No fallback - symbols.csv is required for live trading
        self.logger.error("symbols.csv not found in any expected location. Cannot proceed without universe.")
        self.logger.error(f"Searched paths: {[str(p) for p in possible_paths]}")
        return pd.DataFrame(columns=['symbol'])

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
            except ConnectionError as e:
                self.logger.error(f"Network error getting quotes for batch: {e}")
            except ValueError as e:
                self.logger.error(f"Invalid quote data for batch: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error getting quotes for batch: {e}")
        
        self.logger.info(f"Quotes fetched for {len(quotes)} symbols (requested={len(symbols)})")
        return quotes

    def _get_daily_stats_from_api(self, symbol: str, day: datetime) -> Optional[Dict]:
        """
        Get daily OHLCV from Alpaca API for a specific day.
        
        Returns dict with: 'open', 'close', 'volume'
        """
        try:
            # Get daily bar for the specific day
            start = pd.Timestamp(day).normalize()
            end = start + pd.Timedelta(days=1)
            
            bars = self.api.get_bars(
                symbol, 
                '1Day', 
                start=start.isoformat(),
                end=end.isoformat()
            ).df
            
            if bars.empty:
                return None
            
            # Return OHLCV stats for the day
            last_bar = bars.iloc[-1]
            return {
                'open': float(last_bar['open']),
                'high': float(last_bar['high']),
                'low': float(last_bar['low']),
                'close': float(last_bar['close']),
                'volume': int(last_bar['volume'])
            }
            
        except Exception as e:
            self.logger.debug(f"Error getting daily stats for {symbol} on {day.date()}: {e}")
            return None

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

    def calculate_gaps(self, symbols: List[str]) -> pd.DataFrame:
        """
        Calculate gap percentages for screening (live mode).
        
        Args:
            symbols: List of symbols to calculate gaps for
            
        Returns:
            DataFrame with gap data
        """
        # Ensure gap calculator is initialized
        if self.gap_calculator is None:
            self.get_universe()  # This will initialize gap_calculator
        
        if self.gap_calculator is None:
            self.logger.error("Gap calculator not initialized - no universe loaded")
            return pd.DataFrame()
        
        # Use unified gap calculator
        today = datetime.now()
        result = self.gap_calculator.calculate_gaps(today)
        
        gaps_df = result['gaps']
        
        # Filter to requested symbols only
        if not gaps_df.empty and symbols:
            gaps_df = gaps_df[gaps_df['symbol'].isin(symbols)]
        
        self.logger.info(
            f"Gap calculation complete: {len(gaps_df)} symbols "
            f"(requested={len(symbols)})"
        )
        
        return gaps_df

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
        # Try Alpaca first
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
        except ConnectionError as e:
            self.logger.warning(f"Network error getting Alpaca fundamentals for {symbol}: {e}")
        except ValueError as e:
            self.logger.warning(f"Invalid Alpaca fundamental data for {symbol}: {e}")
        except Exception as e:
            self.logger.debug(f"Alpaca fundamentals unavailable for {symbol}: {e}")
        
        # Fallback to yfinance
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
        except ConnectionError as e:
            self.logger.error(f"Network error getting yfinance fundamentals for {symbol}: {e}")
        except KeyError as e:
            self.logger.warning(f"Missing yfinance data for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error getting yfinance fundamentals for {symbol}: {e}")
        
        return None

    # Add this method to APIDataHandler — it is the only gap vs the Protocol.
    # LocalDataHandler already implements has_data_for_date natively.

    def has_data_for_date(self, date_str: str) -> bool:
        """
        Live mode: data is always available for today (market hours permitting).
        Returns True unconditionally — the live screener does not pre-filter by date.
        """
        return True