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
    MIN_GAP_PERCENT: float = 10.0
    MIN_PRICE: float = 2.00
    # MAX_PRICE removed — upper price filtering is no longer applied.
    # Candidate quality is controlled by MAX_FLOAT, MAX_MARKETCAP, and MIN_GAP_PERCENT.
    MIN_FLOAT: int = 0
    MAX_FLOAT: int = 100_000_000
    MAX_MARKETCAP: float = 2_000_000_000.0
    MIN_RELATIVE_VOLUME: float = 2.0
    ENABLE_RELATIVE_VOLUME: bool = False
    RELATIVE_VOLUME_LOOKBACK_DAYS: int = 10

    # Daily volume pre-screening (from aggregates)
    MIN_DAILY_VOLUME: int = 50_000
    MIN_CUMULATIVE_VOLUME: int = 10_000
    ENABLE_DAILY_VOLUME_PRESCREEN: bool = True

    # Fundamental filters (from aggregates)
    ENABLE_FLOAT_FILTER: bool = True
    ENABLE_MARKETCAP_FILTER: bool = True

    # Cumulative volume tracking (intraday files)
    CUMULATIVE_VOLUME: bool = False

@dataclass(frozen=True)
class GapMonitoringConfig:
    """Gap monitoring frequency configuration for adaptive filtering."""
    # Monitoring intervals based on gap % progress
    NEGATIVE_GAP_INTERVAL_MIN: int = 1  # Negative gap %
    LOW_GAP_INTERVAL_MIN: int = 1       # < 50% of min gap requirement
    MID_GAP_INTERVAL_MIN: int = 1       # 50-90% of min gap requirement
    HIGH_GAP_INTERVAL_MIN: int = 1       # >= 90% of min gap requirement
    QUALIFIED_INTERVAL_MIN: int = 1      # Meets all criteria - pattern analysis active
    
    # Threshold percentages (relative to MIN_GAP_PERCENT)
    LOW_THRESHOLD_PCT: float = 0.50      # 50% of min gap
    HIGH_THRESHOLD_PCT: float = 0.90     # 90% of min gap
    DEQUALIFY_THRESHOLD_PCT: float = 0.75  # Falls below 75% - downgrade monitoring
    
    # Performance tuning
    BATCH_SIZE: int = 500                # Symbols per batch for vectorized operations
    MAX_WORKER_THREADS: int = 4          # Thread pool size for parallel processing
    AGGREGATE_CACHE_MONTHS: int = 2      # Keep N months of aggregates in memory

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
    REGULAR_WARMUP_MINUTES: int = 15
    REGULAR_MIN_BARS: int = 5
    
    # Backtest-specific slippage
    PREMARKET_SLIPPAGE_MULT: float = 2.0
    REGULAR_SLIPPAGE_MULT: float = 1.0

@dataclass(frozen=True)
class PatternConfig:
    """Pattern recognition parameters with dynamic thresholds based on gap size."""
    MIN_STEP_UPS: int = 2
    MIN_ADVANCE_RETENTION: float = 35.0
    MAX_PULLBACK_PERCENT: float = 40.0
    PATTERN_WINDOW_SIZES: List[int] = field(default_factory=lambda: [5, 15, 30])

    # Parabolic pattern thresholds
    PARABOLIC_MIN_ANGLE: float = -999
    PARABOLIC_MAX_ANGLE: float = 999
    PARABOLIC_MIN_ACCELERATION: float = -999.0
    PARABOLIC_MIN_VOL_MULTIPLIER: float = 000
    PARABOLIC_MAX_SCORE: float = 100.0

    # Breakout pattern thresholds
    BREAKOUT_LOOKBACK: int = 20
    BREAKOUT_PRICE_BUFFER_PCT: float = 0.005
    BREAKOUT_VOL_MULTIPLIER: float = 0.5
    
    # Dynamic confluence thresholds based on gap size
    CONFLUENCE_EXTREME_GAP_THRESHOLD: float = 100.0
    CONFLUENCE_EXTREME_GAP_MIN_SCORE: float = 5.0
    
    CONFLUENCE_LARGE_GAP_THRESHOLD: float = 50.0
    CONFLUENCE_LARGE_GAP_MIN_SCORE: float = 10.0
    
    CONFLUENCE_NORMAL_GAP_MIN_SCORE: float = 15.0
    CONFLUENCE_MAX_SCORE: float = 25.0
    CONFLUENCE_MIN_PATTERNS: int = 1
    
    # ✅ NEW: Extreme gap penalty thresholds
    EXTREME_GAP_PENALTY_ENABLED: bool = False
    EXTREME_GAP_3000_THRESHOLD: float = 3000.0  # 3000%+ gap
    EXTREME_GAP_3000_PENALTY: float = 0.5        # 50% penalty (multiply by 0.5)
    
    EXTREME_GAP_2500_THRESHOLD: float = 2500.0  # 2500%+ gap
    EXTREME_GAP_2500_PENALTY: float = 0.7        # 30% penalty (multiply by 0.7)
    
    EXTREME_GAP_2000_THRESHOLD: float = 2000.0  # 2000%+ gap
    EXTREME_GAP_2000_PENALTY: float = 0.9        # 10% penalty (multiply by 0.9)
    
    # Pattern confluence weights (must sum to 1.0 for interpretability)
    CONFLUENCE_WEIGHT_STEP_UP: float = 0.4
    CONFLUENCE_WEIGHT_PARABOLIC: float = 0.0
    CONFLUENCE_WEIGHT_BREAKOUT: float = 0.0
    CONFLUENCE_WEIGHT_VOLUME: float = 0.25
    CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE: float = 0.25

@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters - percentage-based with ATR trailing stops."""
    STOP_LOSS_PERCENT_OF_ACCOUNT: float = 4.0
    MAX_HOLD_TIME_MINUTES: int = 30
    
    # ✅ Break-even threshold for time limit behavior
    BREAKEVEN_THRESHOLD_PCT: float = 0.5  # +/- 0.5% is considered break-even
    ENFORCE_TIME_LIMIT_ON_LOSERS: bool = True  # Force exit losers at time limit (CUT LOSSES)
    
    # ATR-based trailing stop parameters
    ATR_TRAILING_ENABLED: bool = True
    ATR_TRAILING_PERIOD: int = 10
    ATR_TRAILING_MULTIPLIER: float = 1.5
    ATR_TRAILING_MIN_PROFIT_PCT: float = 1.0
    
    # Position sizing and portfolio limits
    MAX_POSITION_SIZE_PERCENT: float = 75.0
    MAX_DAILY_LOSS_PERCENT: float = 6.0
    MAX_CONCURRENT_POSITIONS: int = 3
    MAX_SECTOR_EXPOSURE: float = 0.50
    MAX_DRAWDOWN_PERCENT: float = 20.0
    MAX_PORTFOLIO_HEAT_PERCENT: float = 6.0
    
    # ✅ SLIPPAGE SIMULATION TOGGLE
    ENABLE_SLIPPAGE: bool = False  # Master switch for all slippage simulation
    
    # Winner slippage - reduce profit (exits into bid/ask spread)
    SLIPPAGE_WINNER_MULTIPLIER: float = 0.98  # Keep 98% of profit (2% slippage)
    
    # Loser slippage - increase loss (time exits get worse fills)
    SLIPPAGE_LOSER_MULTIPLIER: float = 1.06  # Losses become 6% worse
    
    # Stop loss slippage - price slips AWAY from entry (most severe)
    SLIPPAGE_STOP_HIGH_GAP_PCT: float = 0.12  # 12% slippage for high gap stocks (>200%)
    SLIPPAGE_STOP_NORMAL_PCT: float = 0.08     # 8% slippage for normal stocks (<=200%)
    SLIPPAGE_GAP_THRESHOLD: float = 200.0      # Gap % threshold for high slippage

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
    LOG_LEVEL: str = 'DEBUG'
    DATABASE_PATH: str = 'trading_system.db'
    BATCH_SIZE: int = 100
    MAX_RETRIES: int = 3
    
    LOG_PROGRESS_EVERY_N_SYMBOLS: int = 10
    LOG_DAILY_SUMMARY: bool = True
    LOG_TRADE_DETAILS: bool = True
    
    # Memory management
    FORCE_GARBAGE_COLLECTION: bool = True
    GC_FREQUENCY_DAYS: int = 1
    MAX_DAY_CACHE_SIZE: int = 2  # Only 2 days (~270 MiB)
    CLEAR_CACHE_BETWEEN_DAYS: bool = True
    USE_FILE_INDEX_CACHE: bool = True  # Enable pre-built file index

    REPORTS_DIR: str = r"S:\trading\reports"  # All backtest outputs go here

@dataclass(frozen=True)
class MarketContextConfig:
    """Shared configuration for MarketContext (live and backtest)."""
    SPY_SYMBOL: str = 'SPY'
    VIX_SYMBOL: str = 'VIX'
    RUT_SYMBOL: str = 'RUT'

    CSV_DIR: str = r'S:\trading\market_context'  # line 217

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
    END_DATE: str = '2024-01-31'
    INITIAL_CAPITAL: float = 1000.0

    # ✅ UPDATED: Simplified paths - daily_aggregates now at root level
    BASE_DATA_DIR: str = r'S:\trading'                          # line 253
    DATA_DIR: str = r'S:\trading\ticker_data'                   # line 254
    NEWS_DATA_DIR: str = r'S:\trading\news_data'                # line 255
    DAILY_AGGREGATES_DIR: str = r'S:\trading\daily_aggregates'  # line 256

    INTRADAY: bool = True
    INTRADAY_TIMEFRAME: str = '1Min'
    ENTRY_CUTOFF_MINUTES: int = 240
    MAX_CANDIDATES_PER_DAY: int = 15
    
    # ✅ SIMPLIFIED NEWS FILTERING
    IGNORE_CATALYST: bool = False           # Enable news filtering
    MAX_NEGATIVE_SENTIMENT: float = 0.08   # Max negative sentiment allowed (0-1 scale)
    MIN_POSITIVE_SENTIMENT: float = 0.0    # reject if max pos < this (0.0 = disabled)

    DATA_TZ: str = "US/Eastern"
    OPEN_WINDOW_MINUTES: int = 15

    AUTO_CREATE_STORAGE_DIRS: bool = False
    ALLOW_FIRST_DAY_WITHOUT_PREV: bool = False

    FAST_MODE: bool = False
    LOG_LEVEL_OVERRIDE: Optional[str] = None
    ENABLE_REENTRY: bool = False
    SIMPLE_STOPS: bool = False
    
    LOG_DAILY_PROGRESS: bool = True
    LOG_SCREENING_DETAILS: bool = True
    LOG_PATTERN_ANALYSIS: bool = True

    ANALYSIS_WINDOW_ENABLED: bool = True
    ANALYSIS_WINDOW_START_ET: str = "06:00"
    ANALYSIS_WINDOW_END_ET: str = "12:00"

@dataclass(frozen=True)
class TradingConfig:
    """Central configuration for the trading system."""
    RUN_MODE: str = 'backtest'   # 'backtest' | 'live'

    api: APIConfig = field(default_factory=APIConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    gap_monitoring: GapMonitoringConfig = field(default_factory=GapMonitoringConfig)
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