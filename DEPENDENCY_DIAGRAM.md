# Paramo Trading System: Dependency Diagram

## Current Architecture (Issues Highlighted)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TradingSystem (main.py)                     │
│                      [TOO MANY RESPONSIBILITIES]                   │
│                                                                     │
│  ├─ Initialization ──────────────────────┐                         │
│  ├─ Schedule Management                 │                         │
│  ├─ Market Scanning                     │                         │
│  ├─ Position Management                 │ [SRP VIOLATION]         │
│  ├─ Risk Monitoring                     │                         │
│  ├─ Reporting                           │                         │
│  └─ Lifecycle Management ───────────────┘                         │
│                                                                     │
│  Direct Dependencies (10+):                                        │
│  ├─ Alpaca REST API ✗ [Direct instantiation]                      │
│  ├─ APIDataHandler                                                │
│  ├─ LiveScreener                                                  │
│  ├─ PatternAnalyzer                                               │
│  ├─ RiskManager                                                   │
│  ├─ TradeExecutor                                                 │
│  ├─ Monitor                                                       │
│  ├─ NewsIntegrationLive                                           │
│  ├─ MarketContext                                                │
│  ├─ Backtester                                                   │
│  └─ NewsIntegrationBacktest                                       │
└─────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ├─────────┬──────────┼──────────┬────────┤
        │         │          │          │        │
        ▼         ▼          ▼          ▼        ▼
    ┌─────┐  ┌──────────┐  ┌────────┐ ┌────┐ ┌──────────┐
    │API  │  │Data      │  │Screener│ │Core│ │Market    │
    │     │  │Handler   │  │        │ │    │ │Context   │
    └─────┘  └──────────┘  └────────┘ └────┘ └──────────┘
        ▲         │           │        │ │         │
        │         │           │        │ │         │
        └─────────┴───────────┴────────┴─┴─────────┘
              [TIGHT COUPLING - Many shared references]

```

## Module Dependencies (Showing Duplication & Inconsistency)

```
CONFIGURATION DUPLICATION
┌──────────────────┐    ┌──────────────────┐
│ /core/config.py  │    │ /config/config.py │
│  (200 lines)     │    │  (205 lines)      │
│ - APIConfig      │    │ - APIConfig       │
│ - ScreeningConfig│    │ - ScreeningConfig │
│ - PatternConfig  │    │ - PatternConfig   │
│ - RiskConfig     │    │ - RiskConfig      │
│ ... [DUPLICATE]  │    │ ... [DUPLICATE]   │
└──────────────────┘    └──────────────────┘
        ▲                        ▲
        │ from config ✗          │ from config.config ✗
        │ INCONSISTENT IMPORTS   │
    ┌───┴──────┬─────────────────┴────┐
    │           │                      │
main.py    trade_executor.py    risk_manager.py
pattern_analyzer.py         [CONF IMPORT PATHS DIFFER]

═══════════════════════════════════════════════════════════

ERROR HANDLING DUPLICATION
┌─────────────────────┐  ┌─────────────────┐  ┌──────────────┐
│ risk_manager.py     │  │ trade_executor  │  │ pattern_     │
│                     │  │ .py             │  │ analyzer.py  │
│ def log_api_error() │  │ def log_api_    │  │ def log_     │
│                     │  │ error() [SAME]  │  │ pattern_     │
│ [IDENTICAL CODE]    │  │                 │  │ error()      │
└─────────────────────┘  └─────────────────┘  └──────────────┘

├─ data_handler/api.py:       log_data_error()
├─ monitor.py:               log_db_error()
└─ All pattern_analyzer.py:  log_pattern_error()
  [5 DIFFERENT IMPLEMENTATIONS OF SAME LOGIC]

═══════════════════════════════════════════════════════════

DATA HANDLER COUPLING
┌──────────────────────┐  ┌────────────────────┐
│  APIDataHandler      │  │  LocalDataHandler  │
│  (Live Trading)      │  │  (Backtesting)     │
├──────────────────────┤  ├────────────────────┤
│ - get_intraday_bars()│  │ - get_intraday_...│
│ - get_universe()     │  │ - get_universe()   │
│ - get_quote_data()   │  │ - get_quote_data() │
│ - calculate_gaps()   │  │ - calculate_gaps() │
│ - get_float_data()   │  │ - get_float_data() │
└──────────────────────┘  └────────────────────┘
         │                         │
         │  NO COMMON             │
         │  INTERFACE             │
         │  ABC ✗                 │
         │                         │
         └────────┬────────────────┘
                  │
         [UNION TYPE USED IN PatternAnalyzer]
         Union[APIDataHandler, LocalDataHandler]
                  │
         [CAN'T INJECT MOCKS EASILY]

═══════════════════════════════════════════════════════════

NEWS INTEGRATION DUPLICATION
┌────────────────────────┐  ┌────────────────────────┐
│  NewsIntegrationLive   │  │  NewsIntegrationBacktest│
│  (Live Trading)        │  │  (Backtesting)         │
├────────────────────────┤  ├────────────────────────┤
│ CATALYST_KEYWORDS {    │  │ CATALYST_KEYWORDS {    │
│   'fda': [...],        │  │   'fda': [...],        │
│   'earnings': [...],   │  │   'earnings': [...],   │
│   ...                  │  │   ...                  │
│ } [DUPLICATE]          │  │ } [DUPLICATE]          │
│                        │  │                        │
│ analyze_news_impact()  │  │ analyze_news_impact()  │
│ get_news_for_symbol()  │  │ get_news_for_symbol()  │
│ _empty_analysis()      │  │ _empty_analysis()      │
└────────────────────────┘  └────────────────────────┘
         │                          │
         │   NO COMMON             │
         │   INTERFACE             │
         │   ABC ✗                 │
         └────────┬────────────────┘
                  │
         [BOTH USED BY BACKTESTER]

═══════════════════════════════════════════════════════════

MARKET CONTEXT DUPLICATION
┌─────────────────────────┐  ┌──────────────────────┐
│  MarketContext (live)   │  │  BacktestMarketContext
│  /market_context/live.py│  │  /market_context/...test
├─────────────────────────┤  ├──────────────────────┤
│ _analyze_spy_trend()    │  │ _spy_trend()         │
│   bars['sma_fast'] =... │  │   bars['sma_fast']...│
│   bars['sma_slow'] =... │  │   bars['sma_slow']...│
│   [100+ DUPLICATE LINES]│  │   [100+ DUPLICATE]   │
│                         │  │                      │
│ _analyze_rut_trend()    │  │ _rut_trend()         │
│   [SAME LOGIC]          │  │   [SAME LOGIC]       │
│                         │  │                      │
│ _get_vix_level()        │  │ _vix_level()         │
│   [OVERLAPPING CODE]    │  │   [OVERLAPPING CODE] │
└─────────────────────────┘  └──────────────────────┘
         │                          │
         └────────┬────────────────┘
                  │
         [BOTH SHARE market_context.scoring FUNCTIONS]
         BUT WITH SEPARATE IMPLEMENTATIONS

═══════════════════════════════════════════════════════════

PATTERN ANALYZER COUPLING
┌────────────────────────────────────────┐
│      PatternAnalyzer                   │
│  (/core/pattern_analyzer.py)          │
├────────────────────────────────────────┤
│ Dependencies:                          │
│ - config: TradingConfig ✗ [Direct]    │
│ - data_handler: Union[            ✗   │
│     APIDataHandler,  [TIGHT COUPLING]  │
│     LocalDataHandler [NO ABSTRACTION] │
│   ]                                   │
│                                       │
│ Responsibilities:                     │
│ ├─ Pattern Analysis                   │
│ ├─ Caching (1000 limit)              │
│ ├─ Multiple detection types          │
│ ├─ Confluence scoring                │
│ └─ Cache statistics tracking         │
└────────────────────────────────────────┘
         │
    [VIOLATES SRP]

═══════════════════════════════════════════════════════════

RISK MANAGER TIGHT COUPLING
┌────────────────────────────────────────┐
│      RiskManager                       │
│  (/core/risk_manager.py)              │
├────────────────────────────────────────┤
│ Dependencies:                          │
│ - config: TradingConfig ✗ [Direct]    │
│ - api: [RAW API OBJECT] ✗ [TIGHT]     │
│                         [COUPLING]     │
│ Pure functions also at module level:  │
│ - calc_atr_stop()   ✗ [Shared but]    │
│ - calc_position_size() [scattered]    │
│ - calc_profit_target() [across]       │
│                    [modules]           │
└────────────────────────────────────────┘
         │
    [CANNOT MOCK API]
    [PURE FUNCTIONS NOT CENTRALIZED]

═══════════════════════════════════════════════════════════

STATE MANAGEMENT PROBLEMS
┌──────────────────────┐
│  Scattered State     │
├──────────────────────┤
│ TradingSystem:       │
│  - running           │
│  - last_scan_time    │
│  - last_heartbeat    │
│                      │
│ TradeExecutor:       │
│  - active_trades     │
│  - trade_history     │
│  - reentry_candidates│
│  - reentry_history   │
│  [MODIFIED FROM]     │
│  [MULTIPLE THREADS]  │
│                      │
│ RiskManager:         │
│  - daily_pnl         │
│  - peak_balance      │
│  - max_drawdown      │
│  [MODIFIED BY BOTH]  │
│  [risk_manager &]    │
│  [trade_executor]    │
│                      │
│ Backtester:          │
│  - positions         │
│  - daily_pnl         │
│  - _day_sentiment    │
│                      │
│ Monitor:             │
│  - metrics           │
│                      │
│ [NO CENTRALIZED     │
│  STATE MANAGEMENT]  │
│ [NO SYNCHRONIZATION]│
└──────────────────────┘
```

## Recommended Architecture (High-Level)

```
┌────────────────────────────────────────────────────────────────┐
│                    Application Layer                           │
│  ┌─────────────────────────┐    ┌──────────────────────────┐  │
│  │ TradingSystemLive       │    │ BacktestRunner           │  │
│  │ (Live Trading           │    │ (Backtesting             │  │
│  │  Orchestrator)          │    │  Orchestrator)           │  │
│  └─────────────────────────┘    └──────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
              │                              │
              ▼                              ▼
┌────────────────────────────────────────────────────────────────┐
│                  Dependency Injection Layer                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ ServiceLocator/Factory                                   │ │
│  │  - Creates components with proper dependencies          │ │
│  │  - Injects abstractions instead of implementations      │ │
│  │  - Manages component lifecycle                          │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────────┐
│                   Domain Layer (Abstractions)                  │
│  ┌──────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ DataHandler ABC  │  │ NewsInt ABC  │  │ StateManager   │  │
│  │ - get_bars()     │  │ - analyze()  │  │ - update_state │  │
│  │ - get_quotes()   │  │ - get_news() │  │ - get_state()  │  │
│  └──────────────────┘  └──────────────┘  └────────────────┘  │
└────────────────────────────────────────────────────────────────┘
              │
       ┌──────┴──────┬──────────┬──────────┬──────────┐
       │             │          │          │          │
       ▼             ▼          ▼          ▼          ▼
    ┌────┐    ┌──────────┐  ┌──────┐  ┌────────┐ ┌──────┐
    │API │    │Screener  │  │Risk  │  │Trade   │ │Pattern
    │Data│    │          │  │Mgr   │  │Exec    │ │Analyzer
    │Handler  │          │  │      │  │        │ │
    └────┘    └──────────┘  └──────┘  └────────┘ └──────┘
    ┌────┐
    │Local│    [All depend on abstractions, not implementations]
    │Data│
    │Handler│
    └────┘
```

---

## Key Issues in Current Dependency Graph

1. **No Abstraction Layer**: All components know about concrete implementations
2. **Bidirectional Dependencies**: Complex web of cross-module references
3. **God Object (TradingSystem)**: Central component coupled to everything
4. **No Inversion of Control**: Components create their own dependencies
5. **Duplicated Implementations**: Same logic implemented in multiple places
6. **Scattered State**: No single source of truth for application state
7. **No Interface Segregation**: Large interfaces with many responsibilities

