# Paramo — Intraday Gap Trading System

Paramo is a Python-based intraday gap trading system for both **live trading** (Alpaca API) and **historical backtesting** (local parquet files). It targets stocks with significant overnight gaps, confirms momentum with multi-detector pattern analysis, and manages risk with ATR-based stops and position sizing.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Folder Structure](#2-folder-structure)
3. [Data Flow](#3-data-flow)
4. [Configuration](#4-configuration)
5. [Entry Points](#5-entry-points)
6. [Data Formats](#6-data-formats)
7. [Key Classes](#7-key-classes)
8. [Tests](#8-tests)
9. [Design Patterns](#9-design-patterns)

---

## 1. System Overview

| Attribute            | Detail                                                  |
|----------------------|---------------------------------------------------------|
| Timeframe            | 1-minute intraday bars                                  |
| Sessions             | Premarket 4:00–9:30 AM ET; Regular 9:30 AM–4:00 PM ET  |
| Strategy             | Gap momentum with multi-pattern confluence confirmation |
| Max candidates/day   | 15                                                      |
| Max concurrent pos.  | 3                                                       |
| Risk model           | ATR trailing stops, percent-of-account position sizing  |
| Backtest data source | Local parquet files                                     |
| Live data source     | Alpaca REST API v2                                      |

---

## 2. Folder Structure

```
Paramo/
├── run/                        # Entry points & orchestrators
│   ├── dispatcher.py           # Mode router (live vs backtest)
│   ├── backtest_entry.py       # Backtest CLI entry + export
│   ├── live_entry.py           # Live trading CLI entry
│   ├── backtest_engine.py      # Backtester class (day-loop orchestrator)
│   └── live_engine.py          # TradingSystem class (live orchestrator)
│
├── config/
│   ├── config.py               # Frozen dataclasses (single source of truth)
│   └── loader.py               # Three-layer override system + validation
│
├── screener/
│   ├── core.py                 # UnifiedScreener — shared live + backtest pipeline
│   ├── backtest.py             # BacktestScreener wrapper
│   ├── live.py                 # LiveScreener wrapper
│   ├── rules.py                # Price / gap / volume filtering rules
│   └── helpers.py              # Relative volume calculators + diagnostics
│
├── strategy/
│   ├── patterns/
│   │   ├── pattern_analyzer.py     # PatternAnalyzer (5 detectors + confluence)
│   │   ├── analyzer_protocol.py    # Protocol for pattern detectors
│   │   ├── analyzer_factory.py     # Factory builder
│   │   ├── claude_pattern_analyzer.py  # Optional Claude AI detector
│   │   └── dual_pattern_analyzer.py    # Fallback detector chain
│   └── risk_manager.py         # calc_atr, calc_atr_stop, position sizing
│
├── data_handler/
│   ├── base.py                 # DataHandler Protocol (PEP 544)
│   ├── local.py                # LocalDataHandler — backtest parquet loader
│   ├── api.py                  # APIDataHandler — Alpaca live data
│   ├── file_index_cache.py     # Fast day-existence index
│   ├── gap/
│   │   └── gap_calculator.py   # Gap % from prev_close
│   └── aggregates/
│       ├── aggregate_handler.py    # Daily OHLCV + float/marketcap data
│       └── aggregate_prefilter.py  # Pre-screen by float/marketcap
│
├── execution/
│   ├── trade_simulator.py      # Signal → position (backtest)
│   ├── exit_simulator.py       # ATR stops, time limits, slippage
│   └── trade_executor.py       # Live order placement via Alpaca
│
├── market_context/
│   ├── backtest.py             # BacktestMarketContext (SPY/VIX/RUT CSVs)
│   ├── live.py                 # Live market context (API)
│   └── scoring.py              # Score → environment classification
│
├── monitoring/
│   ├── reporting.py            # compute_statistics, text/CSV/JSON export
│   └── monitor.py              # Live trade monitoring
│
├── news/
│   ├── backtest.py             # NewsIntegrationBacktest (JSON files)
│   └── live.py                 # NewsIntegrationLive (API)
│
├── utils/
│   ├── logging.py              # get_logger, setup_logging
│   ├── trade_metrics.py        # Timezone helpers, gap type, volume freshness
│   ├── progress.py             # BacktestProgressTracker
│   └── cli_common.py           # Shared CLI argument helpers
│
├── tests/                      # Test suite (pytest)
├── scripts/                    # Ad-hoc analysis scripts
├── reports/                    # Backtest output directory
├── supplemental docs/          # Additional documentation
├── trading_system.db           # SQLite persistence (live mode)
└── run.log                     # Live execution log
```

---

## 3. Data Flow

```
run/dispatcher.py
    │
    ├─── backtest mode ───► run/backtest_entry.py
    │                             │
    │                       Backtester.run_backtest()
    │                             │
    └─── live mode ───────► run/live_entry.py
                                  │
                            TradingSystem.run()

Both modes share the same pipeline below:
─────────────────────────────────────────────────────────────────
1. CONFIG
   build_config()  →  frozen TradingConfig
   (defaults → env vars → CLI overrides → validation)

2. DATA HANDLER  (swappable via Protocol)
   Backtest: LocalDataHandler  →  parquet files
   Live:     APIDataHandler    →  Alpaca REST API

3. DAILY AGGREGATE HANDLER
   AggregateDataHandler  →  daily_aggregates/YYYY/YYYY-MM.parquet
   Provides: prev_close for gap calc, float, marketcap, avg_volume

4. SCREENING PIPELINE  (UnifiedScreener)
   ┌─ Stage 1: Gap detection
   │   gap_pct = (today_open − prev_close) / prev_close × 100
   │   Keep:  gap_pct ≥ MIN_GAP_PERCENT  (default 10 %)
   │
   ├─ Stage 2: Daily volume pre-screen  (optional)
   │   Keep:  daily_volume ≥ MIN_DAILY_VOLUME  (default 50 k)
   │
   ├─ Stage 3: Relative volume filter  (optional)
   │   Keep:  rvol ≥ MIN_RELATIVE_VOLUME  (default 2.0×)
   │
   ├─ Stage 4: Fundamental filters  (optional)
   │   Keep:  float < MAX_FLOAT (100 M)  &  mcap < MAX_MARKETCAP (2 B)
   │
   ├─ Stage 5: Load intraday 1-min bars
   │   Require ≥ warmup bars (30 premarket / 15 regular)
   │
   ├─ Stage 6: Pattern analysis  (PatternAnalyzer)
   │   5 detectors → confluence score → threshold check
   │
   └─ Stage 7: News gate  (optional)
       Sentiment check via NewsIntegration

   Output: List[CandidateSignal]
   {symbol, entry_ts, entry_price, stop_price, gap_percent,
    pattern_strength, relative_volume, meta}

5. TRADE SIMULATION  (TradeSimulator + ExitSimulator)
   ┌─ Pre-entry: position limits, market context adjustment
   ├─ Entry:     ATR stop, percent-of-account sizing
   └─ Exit scan: ATR trailing stop, max hold time, slippage

6. RESULTS
   trades[], equity_curve[], candidate_diagnostics[], statistics{}

7. REPORTING  (monitoring/reporting.py)
   trades.csv · statistics.json · report.txt ·
   candidate_diagnostics.csv · gate_impact.csv
```

---

## 4. Configuration

All configuration lives in **`config/config.py`** as frozen dataclasses. No other config files should be created.

```python
@dataclass(frozen=True)
class TradingConfig:
    RUN_MODE: str = 'backtest'       # 'backtest' | 'live'
    api:            APIConfig
    screening:      ScreeningConfig
    session:        SessionConfig
    pattern:        PatternConfig
    risk:           RiskConfig
    backtest:       BacktestConfig
    market_context: MarketContextConfig
    system:         SystemConfig
    # ... (claude_analyzer, claude_news, gap_monitoring)
```

### Override Priority (lowest → highest)

| Layer | Mechanism | Example |
|-------|-----------|---------|
| 1 — Defaults | Dataclass field defaults | `MIN_GAP_PERCENT = 10.0` |
| 2 — Environment | `PARAMO__section__FIELD=value` | `PARAMO__risk__STOP_LOSS_PERCENT_OF_ACCOUNT=5.0` |
| 3 — CLI | `--override section.FIELD=value` | `--override screening.MIN_GAP_PERCENT=15.0` |

All overrides are applied immutably via `dataclasses.replace()`. Type coercion is automatic (bool/int/float/list inferred from the field's default type).

### Key Config Sections

| Section | Notable Fields |
|---------|---------------|
| `screening` | `MIN_GAP_PERCENT`, `MIN_PRICE`, `ENABLE_FLOAT_FILTER`, `ENABLE_RELATIVE_VOLUME` |
| `pattern` | `CONFLUENCE_WEIGHT_*`, `MIN_STEP_UPS`, `MIN_ADVANCE_RETENTION`, threshold scores |
| `risk` | `STOP_LOSS_PERCENT_OF_ACCOUNT`, `MAX_HOLD_TIME_MINUTES`, `MAX_CONCURRENT_POSITIONS`, `MAX_DAILY_LOSS_PERCENT` |
| `session` | `PREMARKET_ENABLED`, `PREMARKET_START_ET`, `PREMARKET_WARMUP_MINUTES` |
| `backtest` | `START_DATE`, `END_DATE`, `DATA_DIR`, `BASE_DATA_DIR`, `INITIAL_CAPITAL` |
| `system` | `USE_FILE_INDEX_CACHE`, `REPORTS_DIR`, `FORCE_GARBAGE_COLLECTION` |

---

## 5. Entry Points

### Run via dispatcher (recommended)

```bash
python run/dispatcher.py
python run/dispatcher.py --override RUN_MODE=live
```

### Backtest directly

```bash
python run/backtest_entry.py \
  --start 2024-01-03 \
  --end 2024-03-31 \
  --capital 25000 \
  --override screening.MIN_GAP_PERCENT=15.0 \
  --reports-dir reports/ \
  --log-level INFO
```

Output written to `reports/backtest_YYYYMMDD_HHMMSS/`:

| File | Contents |
|------|----------|
| `effective_config.json` | Exact config used (API keys redacted) |
| `trades.csv` | One row per trade |
| `statistics.json` | All performance metrics |
| `report.txt` | Human-readable summary |
| `candidate_diagnostics.csv` | Every screened candidate + rejection reason |
| `backtest_gate_impact.csv` | Which filter gates blocked potential winners |

### Live trading

```bash
export PARAMO__api__ALPACA_API_KEY=your_key
export PARAMO__api__ALPACA_SECRET_KEY=your_secret
python run/live_entry.py --log-level INFO
python run/live_entry.py --dry-run  # no orders placed
```

---

## 6. Data Formats

### Intraday bars — `ticker_data/YYYY/MM/YYYY-MM-DD.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `symbol` | string | Ticker |
| `timestamp` | datetime64[UTC] | Bar open time, UTC-aware |
| `open` | float64 | |
| `high` | float64 | |
| `low` | float64 | |
| `close` | float64 | |
| `volume` | int64 | |
| `count` | int64 | *(optional)* trades in bar |
| `vwap` | float64 | *(optional)* |

### Daily aggregates — `daily_aggregates/YYYY/YYYY-MM.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `symbol` | string | |
| `date` | date | Trading day |
| `open/high/low/close` | float64 | |
| `volume` | int64 | |
| `float` | float64 | *(optional)* shares outstanding |
| `marketcap` | float64 | *(optional)* |
| `avg_volume_10d` | float64 | *(optional)* required for RVOL filter |
| `avg_volume_20d` | float64 | *(optional)* |

---

## 7. Key Classes

| Class | Module | Responsibility |
|-------|--------|---------------|
| `Backtester` | `run/backtest_engine.py` | Day-loop orchestrator for backtesting |
| `TradingSystem` | `run/live_engine.py` | Live trading orchestrator |
| `UnifiedScreener` | `screener/core.py` | 6-stage gap→pattern filtering pipeline |
| `BacktestScreener` | `screener/backtest.py` | Wraps UnifiedScreener for backtest |
| `PatternAnalyzer` | `strategy/patterns/pattern_analyzer.py` | 5-detector confluence pattern scorer |
| `TradeSimulator` | `execution/trade_simulator.py` | Signal → position sizing → exit |
| `ExitSimulator` | `execution/exit_simulator.py` | ATR stops, time limits, slippage |
| `TradeExecutor` | `execution/trade_executor.py` | Live order placement via Alpaca |
| `LocalDataHandler` | `data_handler/local.py` | Parquet file loader (backtest) |
| `APIDataHandler` | `data_handler/api.py` | Alpaca REST data (live) |
| `AggregateDataHandler` | `data_handler/aggregates/aggregate_handler.py` | Daily OHLCV + fundamentals |
| `BacktestMarketContext` | `market_context/backtest.py` | SPY/VIX/RUT regime scoring |

### PatternAnalyzer detectors

| Detector | Config weight field | What it measures |
|----------|--------------------|--------------------|
| Step-up | `CONFLUENCE_WEIGHT_STEP_UP` | 2+ consecutive higher highs, ≥35% advance retained |
| Parabolic | `CONFLUENCE_WEIGHT_PARABOLIC` | Accelerating momentum, angle + volume multiplier |
| Breakout | `CONFLUENCE_WEIGHT_BREAKOUT` | 20-bar range break on volume |
| Volume | `CONFLUENCE_WEIGHT_VOLUME` | Cumulative volume spike vs baseline |
| Support/Resistance | `CONFLUENCE_WEIGHT_SR` | 2+ touches at horizontal level |

Weights must sum to 1.0. Set a weight to 0 to disable that detector entirely.

### Risk parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `STOP_LOSS_PERCENT_OF_ACCOUNT` | 4% | Max risk per trade as % of capital |
| `MAX_CONCURRENT_POSITIONS` | 3 | Open trades at any one time |
| `MAX_DAILY_LOSS_PERCENT` | 6% | Kill switch for the day |
| `MAX_HOLD_TIME_MINUTES` | 30 | Hard time exit |
| `ATR_TRAILING_MULTIPLIER` | 1.5× | Trailing stop = highest − 1.5×ATR(10) |
| `ATR_TRAILING_MIN_PROFIT_PCT` | 1% | Trailing stop activates after +1% |

---

## 8. Tests

```
tests/
├── integration/
│   └── test_backtest_run.py      # Full pipeline on synthetic fixture data
├── screener/
│   └── test_screener_core.py     # UnifiedScreener unit tests
├── execution/
│   └── test_trade_simulator_validation.py
├── strategy/                     # PatternAnalyzer tests
└── infra/                        # Config, logging, monitor tests
```

```bash
pytest tests/ -v                  # All tests
pytest tests/integration/ -v      # Integration tests only
pytest tests/ -k "gap"            # Filter by name
```

179 tests pass as of the current commit.

---

## 9. Design Patterns

### Protocol-based data abstraction (PEP 544)

`data_handler/base.py` defines `DataHandler` as a structural Protocol. `LocalDataHandler` and `APIDataHandler` satisfy it without any inheritance. Subsystems type-hint against `DataHandler`, making it trivial to swap implementations or inject mocks in tests.

### Frozen dataclasses for config

All config sub-objects are `@dataclass(frozen=True)`. Mutation is impossible after construction. Overrides create a new object via `dataclasses.replace()`, leaving the original untouched. This prevents accidental state mutation across subsystems.

### Layered immutable overrides

`config/loader.py` applies overrides in strict priority order and coerces types automatically. The exported `effective_config.json` guarantees exact reproducibility of any backtest run.

### LRU pattern cache

`PatternAnalyzer` caches results in a 512-entry LRU keyed on `(symbol, bars_hash)`. Repeated calls for the same warmup window (common in backtesting) skip all five detectors, significantly reducing CPU time on large date ranges.

### Diagnostic pipeline

Every rejected candidate is recorded with a structured reason string (e.g., `pattern_invalid:low_score(23.8<40.0),step_ups(0<2)`). Post-run analysis exports a gate impact matrix showing which filters blocked the most potential winners, enabling data-driven parameter tuning.
