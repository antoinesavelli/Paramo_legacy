# =====================================================
# config.py - Configuration and Settings
# =====================================================

import os
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import time

@dataclass(frozen=True)
class APIConfig:
    """API configuration settings."""
    ALPACA_API_KEY: str = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY: str = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL: str = 'https://paper-api.alpaca.markets'

@dataclass(frozen=True)
class ScreeningConfig:
    """Screening parameters (unified for all modes and sessions)."""
    MIN_GAP_PERCENT: float = 50.0  
    MIN_PRICE: float = 2.00
    MAX_PRICE: float = 20.00
    MIN_FLOAT: int = 0  # Ignored in backtest
    MAX_FLOAT: int = 100_000_000  # Ignored in backtest
    MIN_RELATIVE_VOLUME: float = 2.0
    MIN_ABSOLUTE_VOLUME: int = 100_000
    ENABLE_RELATIVE_VOLUME: bool = True
    RELATIVE_VOLUME_LOOKBACK_DAYS: int = 10

@dataclass(frozen=True)
class SessionConfig:
    """Session timing parameters (shared across live and backtest)."""
    # Premarket session
    PREMARKET_ENABLED: bool = True
    PREMARKET_START_ET: str = "04:00"
    PREMARKET_END_ET: str = "09:30"
    PREMARKET_WARMUP_MINUTES: int = 30
    PREMARKET_MIN_BARS: int = 5
    
    # Regular session
    REGULAR_START_ET: str = "09:30"
    REGULAR_END_ET: str = "16:00"
    REGULAR_WARMUP_MINUTES: int = 30
    REGULAR_MIN_BARS: int = 5
    
    # Backtest-specific
    PREMARKET_SLIPPAGE_MULT: float = 2.0
    REGULAR_SLIPPAGE_MULT: float = 1.0
    PREMARKET_MIN_SESSION_VOLUME: int = 10000

@dataclass(frozen=True)
class PatternConfig:
    """Pattern recognition parameters with dynamic thresholds based on gap size."""
    MIN_STEP_UPS: int = 2
    MIN_ADVANCE_RETENTION: float = 35.0
    MAX_PULLBACK_PERCENT: float = 40.0
    PATTERN_WINDOW_SIZES: List[int] = field(default_factory=lambda: [5, 15, 30])

    PARABOLIC_MIN_ANGLE: float = -999
    PARABOLIC_MAX_ANGLE: float = 999.0
    PARABOLIC_MIN_ACCELERATION: float = -999
    PARABOLIC_MIN_VOL_MULTIPLIER: float = -999

    BREAKOUT_LOOKBACK: int = 20
    BREAKOUT_PRICE_BUFFER_PCT: float = 0.005
    BREAKOUT_VOL_MULTIPLIER: float = 0.5
    
    # Dynamic confluence thresholds based on gap size
    CONFLUENCE_EXTREME_GAP_THRESHOLD: float = 100.0
    CONFLUENCE_EXTREME_GAP_MIN_SCORE: float = 5.0
    
    CONFLUENCE_LARGE_GAP_THRESHOLD: float = 50.0
    CONFLUENCE_LARGE_GAP_MIN_SCORE: float = 10.0
    
    CONFLUENCE_NORMAL_GAP_MIN_SCORE: float = 15.0
    CONFLUENCE_MIN_PATTERNS: int = 1
    
    # Pattern confluence weights (must sum to 1.0 for interpretability)
    CONFLUENCE_WEIGHT_STEP_UP: float = 0.40
    CONFLUENCE_WEIGHT_PARABOLIC: float = 0.0
    CONFLUENCE_WEIGHT_BREAKOUT: float = 0.15
    CONFLUENCE_WEIGHT_VOLUME: float = 0.20
    CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE: float = 0.25

@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters - percentage-based with ATR trailing stops."""
    STOP_LOSS_PERCENT_OF_ACCOUNT: float = 2.0
    MAX_HOLD_TIME_MINUTES: int = 30
    
    # ATR-based trailing stop parameters
    ATR_TRAILING_ENABLED: bool = True
    ATR_TRAILING_PERIOD: int = 10
    ATR_TRAILING_MULTIPLIER: float = 1.5
    ATR_TRAILING_MIN_PROFIT_PCT: float = 1.0
    
    # Position sizing and portfolio limits
    MAX_POSITION_SIZE_PERCENT: float = 25.0
    MAX_DAILY_LOSS_PERCENT: float = 6.0
    MAX_CONCURRENT_POSITIONS: int = 3
    MAX_SECTOR_EXPOSURE: float = 0.50
    MAX_DRAWDOWN_PERCENT: float = 20.0

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
    
    LOG_PROGRESS_EVERY_N_SYMBOLS: int = 10
    LOG_DAILY_SUMMARY: bool = True
    LOG_TRADE_DETAILS: bool = True
    
    # Memory management
    FORCE_GARBAGE_COLLECTION: bool = True
    GC_FREQUENCY_DAYS: int = 1
    
    # Day cache limit - each day file is ~135 MiB
    MAX_DAY_CACHE_SIZE: int = 2  # Only 2 days (~270 MiB)
    
    CLEAR_CACHE_BETWEEN_DAYS: bool = True

@dataclass(frozen=True)
class MarketContextConfig:
    """Shared configuration for MarketContext (live and backtest)."""
    SPY_SYMBOL: str = 'SPY'
    VIX_SYMBOL: str = 'VIX'
    RUT_SYMBOL: str = 'RUT'

    CSV_DIR: str = r'D:\trading_data\market_context'

    SMA_FAST: int = 5
    SMA_SLOW: int = 20

    SPY_TREND_BULL_WEIGHT: float = 10.0
    SPY_TREND_BEAR_WEIGHT: float = -5.0
    RUT_TREND_BULL_WEIGHT: float = 20.0
    RUT_TREND_BEAR_WEIGHT: float = -15.0
    RUT_LEAD_BONUS: float = 5.0
    RUT_LEAD_MOMENTUM_EDGE: float = 0.5

    VIX_LOW_MAX: float = 15.0
    VIX_NORMAL_MAX: float = 20.0
    VIX_ELEVATED_MAX: float = 30.0
    VIX_HIGH_MAX: float = 50.0

    ENV_FAVORABLE_MIN: float = 70.0
    ENV_NEUTRAL_MIN: float = 40.0
    SHOULD_TRADE_MIN_SCORE_IF_UNFAVORABLE: float = 30.0
    BLOCK_ON_VIX_EXTREME: bool = True

    SIZE_ADJ_FAVORABLE: float = 1.2
    SIZE_ADJ_UNFAVORABLE: float = 0.7

    BREADTH_POSITIVE_RUT_RET_MIN: float = 0.005
    BREADTH_NEGATIVE_RUT_RET_MAX: float = -0.005

@dataclass(frozen=True)
class BacktestConfig:
    """Backtest configuration (intraday only)."""
    START_DATE: str = '2024-01-03'
    END_DATE: str = '2024-08-07'
    INITIAL_CAPITAL: float = 2500.0

    BASE_DATA_DIR: str = r'D:\trading_data'
    DATA_DIR: str = r'D:\trading_data'
    NEWS_DATA_DIR: str = r'D:\trading_data\news'

    INTRADAY: bool = True
    INTRADAY_TIMEFRAME: str = '1Min'
    ENTRY_CUTOFF_MINUTES: int = 240
    MAX_CANDIDATES_PER_DAY: int = 15
    MIN_NEWS_STRENGTH: int = 30
    IGNORE_CATALYST: bool = True

    DATA_TZ: str = "US/Eastern"
    OPEN_WINDOW_MINUTES: int = 15

    AUTO_CREATE_STORAGE_DIRS: bool = False
    ALLOW_FIRST_DAY_WITHOUT_PREV: bool = False

    FAST_MODE: bool = True
    LOG_LEVEL_OVERRIDE: Optional[str] = None
    ENABLE_REENTRY: bool = False
    SIMPLE_STOPS: bool = True
    
    LOG_DAILY_PROGRESS: bool = True
    LOG_SCREENING_DETAILS: bool = True
    LOG_PATTERN_ANALYSIS: bool = False

@dataclass(frozen=True)
class TradingConfig:
    """Central configuration for the trading system."""
    api: APIConfig = field(default_factory=APIConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
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