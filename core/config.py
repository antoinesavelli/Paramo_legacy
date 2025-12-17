# =====================================================
# config.py - Configuration and Settings
# =====================================================

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import time

@dataclass(frozen=True)
class APIConfig:
    """API configuration settings."""
    ALPACA_API_KEY: str = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY: str = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL: str = 'https://paper-api.alpaca.markets'

@dataclass(frozen=True)
class ScreeningConfig:
    """Screening parameters for parabolic moves."""
    MIN_GAP_PERCENT: float = 5.0
    MIN_PRICE: float = 1.00
    MAX_PRICE: float = 20.00
    MIN_MARKET_CAP: float = 25_000_000
    MAX_MARKET_CAP: float = 200_000_000_000
    MIN_FLOAT: int = 0
    MAX_FLOAT: int = 100_000_000
    MIN_RELATIVE_VOLUME: float = 2.0
    MIN_
    
    _VOLUME: int = 200_000
    MAX_SPREAD_PERCENT: float = 2.0
    MIN_DAILY_VOLATILITY: float = 10.0

@dataclass(frozen=True)
class PatternConfig:
    """Pattern recognition parameters."""
    MIN_STEP_UPS: int = 2
    MIN_ADVANCE_RETENTION: float = 60.0
    MAX_PULLBACK_PERCENT: float = 40.0
    MIN_HOLD_TIME_MINUTES: int = 10
    PATTERN_WINDOW_SIZES: List[int] = field(default_factory=lambda: [5, 15, 30])

    # New: make detector thresholds configurable
    PARABOLIC_MIN_ANGLE: float = 45.0
    PARABOLIC_MIN_ACCELERATION: float = 0.02
    PARABOLIC_MIN_VOL_MULTIPLIER: float = 2.5

    BREAKOUT_LOOKBACK: int = 20
    BREAKOUT_PRICE_BUFFER_PCT: float = 0.005  # 0.5% above recent high
    BREAKOUT_VOL_MULTIPLIER: float = 2.0

    # Confluence gate
    CONFLUENCE_MIN_SCORE: float = 55.0
    CONFLUENCE_MIN_PATTERNS: int = 1

@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""
    MAX_RISK_PER_TRADE: float = 25.00
    MAX_POSITION_SIZE_PERCENT: float = 4.0
    MAX_DAILY_LOSS: float = 40.00
    MAX_CONCURRENT_POSITIONS: int = 3
    MAX_SECTOR_EXPOSURE: float = 0.50
    MAX_DRAWDOWN_PERCENT: float = 20.0
    PROFIT_TARGET_CENTS: float = 40.0

@dataclass(frozen=True)
class ReentryConfig:
    """Re-entry parameters."""
    MIN_PULLBACK_FOR_REENTRY: float = 0.10
    MAX_PULLBACK_FOR_REENTRY: float = 0.20
    MAX_REENTRIES_PER_STOCK: int = 3

@dataclass(frozen=True)
class MarketHoursConfig:
    """Market hours (Eastern Time)."""
    PRE_MARKET_START: time = time(4, 0)
    MARKET_OPEN: time = time(9, 30)
    MARKET_CLOSE: time = time(16, 0)
    AFTER_HOURS_END: time = time(20, 0)

@dataclass(frozen=True)
class SystemConfig:
    """System parameters."""
    SCAN_INTERVAL_SECONDS: int = 30
    DATA_UPDATE_INTERVAL: int = 1
    MAX_API_RETRIES: int = 3
    API_RETRY_DELAY: int = 2
    LOG_LEVEL: str = 'INFO'
    DATABASE_PATH: str = 'trading_system.db'
    BATCH_SIZE: int = 100
    MAX_RETRIES: int = 3

@dataclass(frozen=True)
class MarketContextConfig:
    """Shared configuration for MarketContext (live and backtest)."""
    # Symbols
    SPY_SYMBOL: str = 'SPY'
    VIX_SYMBOL: str = 'VXX'
    RUT_SYMBOL: str = 'RUT'  # or 'IWM' for live if needed

    # CSV backtest directory (used by BacktestMarketContext)
    CSV_DIR: str = r'D:\trading_data\market_context'

    # Trend settings
    SMA_FAST: int = 5
    SMA_SLOW: int = 20

    # Weights for scoring (small-cap focused)
    SPY_TREND_BULL_WEIGHT: float = 10.0
    SPY_TREND_BEAR_WEIGHT: float = -5.0
    RUT_TREND_BULL_WEIGHT: float = 20.0
    RUT_TREND_BEAR_WEIGHT: float = -15.0
    RUT_LEAD_BONUS: float = 5.0            # bonus if RUT leads SPY
    RUT_LEAD_MOMENTUM_EDGE: float = 0.5    # momentum edge in percentage points

    # VIX classification bands
    VIX_LOW_MAX: float = 15.0
    VIX_NORMAL_MAX: float = 20.0
    VIX_ELEVATED_MAX: float = 30.0
    VIX_HIGH_MAX: float = 50.0

    # Scoring/environment thresholds
    ENV_FAVORABLE_MIN: float = 70.0
    ENV_NEUTRAL_MIN: float = 40.0
    SHOULD_TRADE_MIN_SCORE_IF_UNFAVORABLE: float = 30.0
    BLOCK_ON_VIX_EXTREME: bool = True

    # Position size adjustments
    SIZE_ADJ_FAVORABLE: float = 1.2
    SIZE_ADJ_UNFAVORABLE: float = 0.7

    # Breadth proxy thresholds (RUT vs SPY daily returns) - already used in BT
    BREADTH_POSITIVE_RUT_RET_MIN: float = 0.005
    BREADTH_NEGATIVE_RUT_RET_MAX: float = -0.005

@dataclass(frozen=True)
class BacktestSessionConfig:
    """Configuration for backtest sessions."""
    PREMARKET_ENABLED: bool = True
    PREMARKET_START_ET: str = "04:00"         # local exchange time
    PREMARKET_END_ET: str = "09:30"
    PREMARKET_WARMUP_MINUTES: int = 25
    PREMARKET_MIN_GAP_PERCENT: float = 3.0
    PREMARKET_MIN_ABSOLUTE_VOLUME: int = 15000
    PREMARKET_MIN_SESSION_VOLUME: int = 10000    # total premarket volume before eligibility
    PREMARKET_MAX_SPREAD_PERCENT: float = 4.0
    PREMARKET_MIN_BARS: int = 15
    # Slippage multiplier relative to regular session
    PREMARKET_SLIPPAGE_MULT: float = 2.0

@dataclass(frozen=True)
class BacktestConfig:
    """Backtest configuration (intraday only)."""
    START_DATE: str = '2025-01-02'
    END_DATE: str   = '2025-01-08'
    INITIAL_CAPITAL: float = 1000.0

    BASE_DATA_DIR: str = r'D:\trading_data'
    DATA_DIR: str = BASE_DATA_DIR
    NEWS_DATA_DIR: str = os.path.join(BASE_DATA_DIR, 'news')
    SENTIMENT_FILE: str = os.path.join(BASE_DATA_DIR, 'sentiment', 'processed')

    INTRADAY: bool = True
    INTRADAY_TIMEFRAME: str = '1Min'
    WARMUP_MINUTES: int = 15
    ENTRY_CUTOFF_MINUTES: int = 240
    MAX_CANDIDATES_PER_DAY: int = 15
    MIN_NEWS_STRENGTH: int = 0
    MIN_MARKET_SENTIMENT: int = 0
    IGNORE_CATALYST: bool = True

    DATA_TZ: str = "US/Eastern"
    OPEN_WINDOW_MINUTES: int = 15

    # Disable auto-creation of folders by default
    AUTO_CREATE_STORAGE_DIRS: bool = False
    ALLOW_FIRST_DAY_WITHOUT_PREV: bool = True
    SESSION: BacktestSessionConfig = field(default_factory=BacktestSessionConfig)

    # New fields for performance
    FAST_MODE: bool = True  # Enable fast backtesting optimizations
    LOG_LEVEL_OVERRIDE: Optional[str] = "WARNING"  # Override log level for backtest
    ENABLE_REENTRY: bool = False  # Disable re-entry logic for speed
    SIMPLE_STOPS: bool = True  # Use fixed percentage stops

@dataclass(frozen=True)
class TradingConfig:
    """Central configuration for the trading system."""
    api: APIConfig = field(default_factory=APIConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    reentry: ReentryConfig = field(default_factory=ReentryConfig)
    market_hours: MarketHoursConfig = field(default_factory=MarketHoursConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    market_context: MarketContextConfig = field(default_factory=MarketContextConfig)

if __name__ == "__main__":
    config = TradingConfig()
    print(config)
    input("Press Enter to exit...")