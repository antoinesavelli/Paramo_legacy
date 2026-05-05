# =====================================================
# config/config.py - Single source of truth for all configuration.
#
# RULES:
#   - This is the ONLY config file in the project. Do NOT create any other
#     config files anywhere in the codebase (e.g. core/, screener/, scripts/).
#   - All new settings belong in the appropriate dataclass below.
#   - All subsystems must obtain config via config.loader.build_config().
# =====================================================

import os
from dataclasses import dataclass, field
from typing import List, Optional

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

    # Absolute volume gate (used by screener/rules.py ScreeningConfigView)
    MIN_ABSOLUTE_VOLUME: int = 50_000

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
    NEGATIVE_GAP_INTERVAL_MIN: int = 1
    LOW_GAP_INTERVAL_MIN: int = 1
    MID_GAP_INTERVAL_MIN: int = 1
    HIGH_GAP_INTERVAL_MIN: int = 1
    QUALIFIED_INTERVAL_MIN: int = 1

    LOW_THRESHOLD_PCT: float = 0.50
    HIGH_THRESHOLD_PCT: float = 0.90
    DEQUALIFY_THRESHOLD_PCT: float = 0.75

    BATCH_SIZE: int = 500
    MAX_WORKER_THREADS: int = 4
    AGGREGATE_CACHE_MONTHS: int = 2

@dataclass(frozen=True)
class SessionConfig:
    """
    Session timing parameters — shared across live and backtest.
    Single source of truth for all session boundaries and market hours.
    (MarketHoursConfig was removed; use these string fields everywhere.)
    """
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

    # After-hours end (used as the upper cap for is_market_hours checks)
    AFTER_HOURS_END_ET: str = "20:00"

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

    # Parabolic pattern thresholds (weight = 0.0 → detector runs but contributes nothing)
    PARABOLIC_MIN_ANGLE: float = -999
    PARABOLIC_MAX_ANGLE: float = 999
    PARABOLIC_MIN_ACCELERATION: float = -999.0
    PARABOLIC_MIN_VOL_MULTIPLIER: float = 0.0
    PARABOLIC_MAX_SCORE: float = 100.0

    # Breakout pattern thresholds (weight = 0.0)
    BREAKOUT_LOOKBACK: int = 20
    BREAKOUT_PRICE_BUFFER_PCT: float = 0.005
    BREAKOUT_VOL_MULTIPLIER: float = 0.5

    # Dynamic confluence thresholds based on gap size
    CONFLUENCE_EXTREME_GAP_THRESHOLD: float = 100.0
    CONFLUENCE_EXTREME_GAP_MIN_SCORE: float = 15.0
    CONFLUENCE_LARGE_GAP_THRESHOLD: float = 50.0
    CONFLUENCE_LARGE_GAP_MIN_SCORE: float = 25.0
    CONFLUENCE_NORMAL_GAP_MIN_SCORE: float = 40.0
    CONFLUENCE_MAX_SCORE: float = 100.0

    # Pattern confluence weights (must sum to 1.0)
    CONFLUENCE_WEIGHT_STEP_UP: float = 0.50
    CONFLUENCE_WEIGHT_PARABOLIC: float = 0.00
    CONFLUENCE_WEIGHT_BREAKOUT: float = 0.00
    CONFLUENCE_WEIGHT_VOLUME: float = 0.25
    CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE: float = 0.25

    CONFLUENCE_MIN_PATTERNS: int = 1

    # Extreme gap score penalty (applied after normalization)
    EXTREME_GAP_PENALTY_ENABLED: bool = False
    EXTREME_GAP_2000_THRESHOLD: float = 2000.0
    EXTREME_GAP_2000_PENALTY: float = 0.85
    EXTREME_GAP_2500_THRESHOLD: float = 2500.0
    EXTREME_GAP_2500_PENALTY: float = 0.75
    EXTREME_GAP_3000_THRESHOLD: float = 3000.0
    EXTREME_GAP_3000_PENALTY: float = 0.60

@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters — percentage-based with ATR trailing stops."""
    STOP_LOSS_PERCENT_OF_ACCOUNT: float = 4.0
    MAX_HOLD_TIME_MINUTES: int = 30

    BREAKEVEN_THRESHOLD_PCT: float = 0.5
    ENFORCE_TIME_LIMIT_ON_LOSERS: bool = True

    # ATR-based trailing stop
    ATR_TRAILING_ENABLED: bool = True
    ATR_TRAILING_PERIOD: int = 10
    ATR_TRAILING_MULTIPLIER: float = 1.5
    ATR_TRAILING_MIN_PROFIT_PCT: float = 1.0

    # Position sizing and portfolio limits
    MAX_POSITION_SIZE_PERCENT: float = 75.0
    MAX_DAILY_LOSS_PERCENT: float = 6.0
    MAX_CONCURRENT_POSITIONS: int = 3
    MAX_SECTOR_EXPOSURE: float = 0.50        # strategy/risk_manager.py:378
    MAX_DRAWDOWN_PERCENT: float = 20.0       # strategy/risk_manager.py:262
    MAX_PORTFOLIO_HEAT_PERCENT: float = 6.0  # strategy/risk_manager.py (portfolio heat gate)

    # Slippage simulation (backtest only)
    ENABLE_SLIPPAGE: bool = False
    SLIPPAGE_WINNER_MULTIPLIER: float = 0.98
    SLIPPAGE_LOSER_MULTIPLIER: float = 1.06
    SLIPPAGE_STOP_HIGH_GAP_PCT: float = 0.12
    SLIPPAGE_STOP_NORMAL_PCT: float = 0.08
    SLIPPAGE_GAP_THRESHOLD: float = 200.0

@dataclass(frozen=True)
class SystemConfig:
    """System / infrastructure parameters."""
    SCAN_INTERVAL_SECONDS: int = 30
    MAX_API_RETRIES: int = 3
    API_RETRY_DELAY: int = 2
    LOG_LEVEL: str = 'DEBUG'
    MAX_RETRIES: int = 3

    DATABASE_PATH: str = 'trading_system.db'  # core/monitor.py:22
    BATCH_SIZE: int = 100                      # data_handler/api.py:95

    LOG_PROGRESS_EVERY_N_SYMBOLS: int = 10
    LOG_DAILY_SUMMARY: bool = True
    LOG_TRADE_DETAILS: bool = True

    # Memory management
    FORCE_GARBAGE_COLLECTION: bool = True
    GC_FREQUENCY_DAYS: int = 1
    MAX_DAY_CACHE_SIZE: int = 2
    CLEAR_CACHE_BETWEEN_DAYS: bool = True
    USE_FILE_INDEX_CACHE: bool = True

    REPORTS_DIR: str = r"S:\trading\reports"

@dataclass(frozen=True)
class MarketContextConfig:
    """Shared configuration for MarketContext (live and backtest)."""
    SPY_SYMBOL: str = 'SPY'
    VIX_SYMBOL: str = 'VIX'
    RUT_SYMBOL: str = 'RUT'

    CSV_DIR: str = r'S:\trading\market_context'

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
    """Backtest configuration (intraday 1-min bars)."""
    START_DATE: str = '2024-01-03'
    END_DATE: str = '2024-01-31'
    INITIAL_CAPITAL: float = 1000.0

    BASE_DATA_DIR: str = r'S:\trading'
    DATA_DIR: str = r'S:\trading\ticker_data'
    NEWS_DATA_DIR: str = r'S:\trading\news_data'
    DAILY_AGGREGATES_DIR: str = r'S:\trading\daily_aggregates'

    INTRADAY: bool = True
    INTRADAY_TIMEFRAME: str = '1Min'
    MAX_CANDIDATES_PER_DAY: int = 15
    # ENTRY_CUTOFF_MINUTES removed — ANALYSIS_WINDOW_END_ET is the single entry deadline

    IGNORE_CATALYST: bool = False
    MAX_NEGATIVE_SENTIMENT: float = 0.08
    MIN_POSITIVE_SENTIMENT: float = 0.0

    DATA_TZ: str = "US/Eastern"

    AUTO_CREATE_STORAGE_DIRS: bool = False
    ALLOW_FIRST_DAY_WITHOUT_PREV: bool = False

    FAST_MODE: bool = False
    LOG_LEVEL_OVERRIDE: Optional[str] = None

    # Re-entry: allow a second entry on the same stock after the initial position closes.
    # Set True to enable; wire into screener/core.py and core/trade_executor.py.
    ENABLE_REENTRY: bool = False

    # SIMPLE_STOPS: use fixed-% stop instead of ATR trailing (backtester/exit_simulator.py:54)
    SIMPLE_STOPS: bool = False

    LOG_DAILY_PROGRESS: bool = True
    LOG_SCREENING_DETAILS: bool = True
    LOG_PATTERN_ANALYSIS: bool = True

    ANALYSIS_WINDOW_ENABLED: bool = True
    ANALYSIS_WINDOW_START_ET: str = "04:00"
    ANALYSIS_WINDOW_END_ET: str = "16:00"

@dataclass(frozen=True)
class _ClaudeBaseConfig:
    """Shared Claude toggle structure for all Claude-backed components."""
    ENABLED: bool = False
    MODE: str = 'hard_coded_only'
    CONSENSUS: str = 'and'
    ENABLED_IN_BACKTEST: bool = False
    TIMEOUT_SECONDS: float = 10.0
    MODEL: str = 'claude-sonnet-4-5'

@dataclass(frozen=True)
class ClaudeAnalyzerConfig(_ClaudeBaseConfig):
    """Claude config for OHLCV pattern analysis."""
    MAX_BARS_TO_SEND: int = 60

@dataclass(frozen=True)
class ClaudeNewsConfig(_ClaudeBaseConfig):
    """Claude config for news sentiment scoring."""
    MAX_ARTICLES_TO_SEND: int = 5
    FETCH_ARTICLE_BODY: bool = True

@dataclass(frozen=True)
class TradingConfig:
    """
    Root configuration object. Instantiate via config.loader.build_config().
    Do NOT import sub-dataclasses directly into production code.
    """
    RUN_MODE: str = 'backtest'   # 'backtest' | 'live'

    api: APIConfig = field(default_factory=APIConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    gap_monitoring: GapMonitoringConfig = field(default_factory=GapMonitoringConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    market_context: MarketContextConfig = field(default_factory=MarketContextConfig)
    claude_analyzer: ClaudeAnalyzerConfig = field(default_factory=ClaudeAnalyzerConfig)
    claude_news: ClaudeNewsConfig = field(default_factory=ClaudeNewsConfig)

if __name__ == "__main__":
    import json
    from dataclasses import asdict
    cfg = TradingConfig()
    print(json.dumps(asdict(cfg), indent=2, default=str))