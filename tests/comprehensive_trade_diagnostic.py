# =====================================================
# comprehensive_trade_diagnostic.py - In-Depth Backtest Diagnostic Tool
# =====================================================
"""
Comprehensive diagnostic script that analyzes the entire screening and pattern
analysis pipeline to understand why stocks that previously traded are now filtered out.

This script provides detailed insights into:
- Stocks detected in the universe
- Gap calculations and thresholds
- Screening filter results (price, gap, relative volume)
- Warmup bar requirements and rejections
- Pattern analysis failures (step-up, parabolic, breakout, etc.)
- Configuration settings that affect each stage

PERFORMANCE OPTIMIZATION:
- Analyzes ALL symbols (no sampling)
- Limited to January 2024 timeframe only
- Processes full universe but reduced date range
""" 

import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import Counter, defaultdict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.loader import build_config
from data_handler.local import LocalDataHandler
from strategy.pattern_analyzer import PatternAnalyzer
from screener.backtest import BacktestScreener
from screener.rules import cfg_view_from
from utils.logging import get_logger


class ComprehensiveTradeDiagnostic:
    """Comprehensive diagnostic tool for analyzing the entire trading pipeline."""
    
    def __init__(self, config):
        """
        Initialize diagnostic tool.
        
        Args:
            config: TradingConfig object
        """
        self.config = config
        self.logger = get_logger(__name__, component="comprehensive_diagnostic")
        
        self.data_handler = LocalDataHandler(
            config, 
            data_dir=config.backtest.DATA_DIR
        )
        
        self.pattern_analyzer = PatternAnalyzer(config, self.data_handler)
        self.screener = BacktestScreener(config, self.data_handler, self.pattern_analyzer)
        
        # Results storage
        self.results = {
            'daily_summaries': [],
            'stock_details': [],
            'filter_stats': defaultdict(Counter),
            'pattern_failures': defaultdict(list),
            'warmup_rejections': [],
            'gap_conversion_losses': [],
            'config_snapshot': {}
        }
    
    def analyze_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Analyze complete pipeline for a date range.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with comprehensive diagnostic results
        """
        print("\n" + "=" * 100)
        print("COMPREHENSIVE TRADE DIAGNOSTIC - JANUARY 2024")
        print("=" * 100)
        
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            print(f"✓ Parsed dates: {start.date()} to {end.date()}")
        except Exception as e:
            print(f"✗ Failed to parse dates: {e}")
            raise
        
        self.logger.info("=" * 100)
        self.logger.info(f"COMPREHENSIVE TRADE DIAGNOSTIC - JANUARY 2024")
        self.logger.info(f"Date Range: {start_date} to {end_date}")
        self.logger.info(f"Mode: Full universe, January 2024 only")
        self.logger.info("=" * 100)
        
        # Capture configuration snapshot
        self._capture_config_snapshot()
        
        # Process each day
        current = start
        days_processed = 0
        
        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                day_str = current.strftime('%Y-%m-%d')
                
                if self.data_handler.has_data_for_date(day_str):
                    self.logger.info("")
                    self.logger.info("=" * 100)
                    self.logger.info(f"ANALYZING: {day_str}")
                    self.logger.info("=" * 100)
                    
                    self._analyze_single_day(current)
                    days_processed += 1
                else:
                    self.logger.info(f"SKIP: {day_str} - No data available")
        
            current += timedelta(days=1)
        
        self.logger.info("")
        self.logger.info("=" * 100)
        self.logger.info(f"ANALYSIS COMPLETE - Processed {days_processed} days")
        self.logger.info(f"Daily summaries stored: {len(self.results['daily_summaries'])}")
        self.logger.info("=" * 100)
        
        # Generate summary report
        return self._generate_comprehensive_report()
    
    def _analyze_single_day(self, day: datetime):
        """Analyze complete pipeline for a single trading day."""
        day_str = day.strftime('%Y-%m-%d')
        
        daily_summary = {
            'date': day_str,
            'stage': {}
        }
        
        # STAGE 1: Universe Detection
        self.logger.info("[STAGE 1] UNIVERSE DETECTION")
        universe_stats = self._analyze_universe(day)
        daily_summary['stage']['universe'] = universe_stats
        
        # STAGE 2: Gap Calculation
        self.logger.info("[STAGE 2] GAP CALCULATION")
        gap_stats = self._analyze_gaps_new(day)
        daily_summary['stage']['gaps'] = gap_stats
        
        # STAGE 3: Screening Filters
        self.logger.info("[STAGE 3] SCREENING FILTERS")
        screening_stats = self._analyze_screening_filters(day, gap_stats.get('gaps_df'))
        daily_summary['stage']['screening'] = screening_stats
        
        # STAGE 3.5: Warmup Bar Analysis
        self.logger.info("[STAGE 3.5] WARMUP BAR ANALYSIS")
        warmup_stats = self._analyze_warmup_requirements(day, screening_stats.get('passed_stocks', []))
        daily_summary['stage']['warmup'] = warmup_stats
        
        # STAGE 4: Pattern Analysis
        self.logger.info("[STAGE 4] PATTERN ANALYSIS")
        pattern_stats = self._analyze_patterns(day, warmup_stats.get('passed_warmup', []))
        daily_summary['stage']['patterns'] = pattern_stats
        
        # STAGE 5: Position Sizing & Risk
        self.logger.info("[STAGE 5] POSITION SIZING & RISK")
        risk_stats = self._analyze_risk_checks(day, pattern_stats.get('valid_patterns', []))
        daily_summary['stage']['risk'] = risk_stats
        
        # Store daily summary
        self.results['daily_summaries'].append(daily_summary)
        
        # Log daily summary
        self._log_daily_summary(daily_summary)
    
    def _analyze_universe(self, day: datetime) -> Dict:
        """Analyze universe detection stage."""
        day_str = day.strftime('%Y-%m-%d')
        
        universe = self.data_handler.get_universe()
        total_symbols = len(universe)
        
        # Check which symbols have data for this day (sample for performance)
        symbols_with_data = []
        symbols_without_data = []
        
        sample_size = min(100, total_symbols)
        for symbol in universe['symbol'].tolist()[:sample_size]:
            day_data = self.data_handler.get_symbol_day_data(symbol, day_str)
            if not day_data.empty:
                symbols_with_data.append(symbol)
            else:
                symbols_without_data.append(symbol)
        
        stats = {
            'total_universe': total_symbols,
            'sampled': sample_size,
            'with_data': len(symbols_with_data),
            'without_data': len(symbols_without_data),
            'sample_symbols': symbols_with_data[:10]
        }
        
        self.logger.info(f"  • Total Universe: {total_symbols:,} symbols")
        self.logger.info(f"  • Sampled: {stats['sampled']} symbols")
        self.logger.info(f"  • With Data: {stats['with_data']} | Without Data: {stats['without_data']}")
        
        return stats
    
    def _analyze_gaps_new(self, day: datetime) -> Dict:
        """Analyze gap calculation using aggregate handler."""
        from data_handler.aggregate_handler import AggregateDataHandler
        from pathlib import Path
        
        # Initialize aggregate handler
        agg_dir = Path(self.config.backtest.BASE_DATA_DIR) / "daily_aggregates"
        agg_handler = AggregateDataHandler(str(agg_dir))
        
        # Get current day aggregates
        day_agg = agg_handler.get_day_aggregates(day)
        
        if day_agg is None or day_agg.empty:
            self.logger.warning(f"No aggregate data for {day.strftime('%Y-%m-%d')}")
            return {
                'total_gaps': 0,
                'gaps_df': pd.DataFrame(),
                'prev_day_used': 'N/A'
            }
        
        # Get previous day
        prev_day = day - timedelta(days=1)
        lookback = 0
        while lookback < 7:
            prev_agg = agg_handler.get_day_aggregates(prev_day)
            if prev_agg is not None and not prev_agg.empty:
                break
            prev_day = prev_day - timedelta(days=1)
            lookback += 1
        
        if prev_agg is None or prev_agg.empty:
            self.logger.warning(f"No previous day data found within 7 days")
            return {
                'total_gaps': 0,
                'gaps_df': pd.DataFrame(),
                'prev_day_used': 'N/A',
                'lookback_days': lookback
            }
        
        # Calculate gaps
        common_symbols = set(day_agg['symbol']) & set(prev_agg['symbol'])
        
        gaps_list = []
        for symbol in common_symbols:
            day_row = day_agg[day_agg['symbol'] == symbol].iloc[0]
            prev_row = prev_agg[prev_agg['symbol'] == symbol].iloc[0]
            
            gap_pct = ((day_row['open'] - prev_row['close']) / prev_row['close']) * 100.0
            
            gaps_list.append({
                'symbol': symbol,
                'open_price': day_row['open'],
                'prev_close': prev_row['close'],
                'last_price': day_row['close'],
                'gap_percent': gap_pct,
                'volume': day_row['volume']
            })
        
        gaps_df = pd.DataFrame(gaps_list)
        
        stats = {
            'prev_day_used': prev_day.strftime('%Y-%m-%d'),
            'lookback_days': lookback,
            'total_gaps': len(gaps_df),
            'gaps_df': gaps_df
        }
        
        if not gaps_df.empty:
            stats['gap_distribution'] = {
                'min': float(gaps_df['gap_percent'].min()),
                'max': float(gaps_df['gap_percent'].max()),
                'mean': float(gaps_df['gap_percent'].mean()),
                'median': float(gaps_df['gap_percent'].median())
            }
            
            # Gap ranges
            stats['gap_ranges'] = {
                '0-5%': len(gaps_df[gaps_df['gap_percent'].abs() < 5]),
                '5-10%': len(gaps_df[(gaps_df['gap_percent'].abs() >= 5) & (gaps_df['gap_percent'].abs() < 10)]),
                '10-20%': len(gaps_df[(gaps_df['gap_percent'].abs() >= 10) & (gaps_df['gap_percent'].abs() < 20)]),
                '20-50%': len(gaps_df[(gaps_df['gap_percent'].abs() >= 20) & (gaps_df['gap_percent'].abs() < 50)]),
                '50%+': len(gaps_df[gaps_df['gap_percent'].abs() >= 50])
            }
            
            # Top gapper
            top_gap = gaps_df.loc[gaps_df['gap_percent'].idxmax()]
            stats['top_gapper'] = {
                'symbol': top_gap['symbol'],
                'gap_percent': float(top_gap['gap_percent']),
                'last_price': float(top_gap['last_price'])
            }
        
        self.logger.info(f"  • Previous Day: {stats['prev_day_used']} ({stats['lookback_days']} days back)")
        self.logger.info(f"  • Total Gaps Calculated: {stats['total_gaps']}")
        
        if 'gap_distribution' in stats:
            self.logger.info(f"  • Gap Range: {stats['gap_distribution']['min']:.2f}% to {stats['gap_distribution']['max']:.2f}%")
            self.logger.info(f"  • Gap Breakdown:")
            for range_name, count in stats['gap_ranges'].items():
                self.logger.info(f"    - {range_name}: {count} stocks")
        
        return stats
    
    def _analyze_screening_filters(self, day: datetime, gaps_df: Optional[pd.DataFrame]) -> Dict:
        """Analyze screening filter stage."""
        if gaps_df is None or gaps_df.empty:
            return {
                'passed_stocks': [],
                'filter_results': {},
                'message': 'No gaps to analyze'
            }
        
        cfg_screening = self.config.screening
        
        # Track each filter stage
        filter_results = {
            'initial': len(gaps_df),
            'after_gap_threshold': 0,
            'after_price_range': 0,
            'after_volume': 0,
            'after_relative_volume': 0
        }
        
        # Filter 1: Gap Threshold
        min_gap = cfg_screening.MIN_GAP_PERCENT
        after_gap = gaps_df[gaps_df['gap_percent'].abs() >= min_gap]
        filter_results['after_gap_threshold'] = len(after_gap)
        
        self.logger.info(f"  • [FILTER 1] Gap Threshold (>= {min_gap}%)")
        self.logger.info(f"    Before: {filter_results['initial']} | After: {filter_results['after_gap_threshold']} | Filtered: {filter_results['initial'] - filter_results['after_gap_threshold']}")
        
        # Filter 2: Price Range
        after_price = after_gap[
            (after_gap['last_price'] >= cfg_screening.MIN_PRICE) &
            (after_gap['last_price'] <= cfg_screening.MAX_PRICE)
        ]
        filter_results['after_price_range'] = len(after_price)
        
        self.logger.info(f"  • [FILTER 2] Price Range (${cfg_screening.MIN_PRICE:.2f} - ${cfg_screening.MAX_PRICE:.2f})")
        self.logger.info(f"    Before: {filter_results['after_gap_threshold']} | After: {filter_results['after_price_range']} | Filtered: {filter_results['after_gap_threshold'] - filter_results['after_price_range']}")
        
        # Filter 3: Daily Volume (uses aggregate data)
        min_daily_vol = cfg_screening.MIN_DAILY_VOLUME
        
        after_volume = after_price[after_price['volume'] >= min_daily_vol]
        filter_results['after_volume'] = len(after_volume)
        
        self.logger.info(f"  • [FILTER 3] Daily Volume (>= {min_daily_vol:,}) [Using Aggregate Data]")
        self.logger.info(f"    Before: {filter_results['after_price_range']} | After: {filter_results['after_volume']} | Filtered: {filter_results['after_price_range'] - filter_results['after_volume']}")
        
        # Convert to list of dicts
        stocks_with_volume = []
        for _, row in after_volume.iterrows():
            stocks_with_volume.append({
                'symbol': row['symbol'],
                'gap_percent': row['gap_percent'],
                'last_price': row['last_price'],
                'volume': row['volume']
            })
        
        passed_stocks = stocks_with_volume
        filter_results['after_relative_volume'] = filter_results['after_volume']
        
        # Log rejected stocks
        rejected_by_volume = after_price[after_price['volume'] < min_daily_vol]
        if not rejected_by_volume.empty and len(rejected_by_volume) <= 20:
            self.logger.info(f"  • Stocks Rejected by Volume Filter ({len(rejected_by_volume)} stocks):")
            for _, stock in rejected_by_volume.nlargest(10, 'volume').iterrows():
                self.logger.info(f"    - {stock['symbol']}: {stock['volume']:,} volume | Gap: {stock['gap_percent']:+.2f}% | Price: ${stock['last_price']:.2f}")
        
        # Log top survivors
        if passed_stocks:
            self.logger.info(f"  • Top Stocks After Screening:")
            for stock in sorted(passed_stocks, key=lambda x: x['gap_percent'], reverse=True)[:5]:
                self.logger.info(f"    - {stock['symbol']}: {stock['gap_percent']:+.2f}% | ${stock['last_price']:.2f} | Vol: {stock['volume']:,}")
        
        return {
            'passed_stocks': passed_stocks,
            'filter_results': filter_results
        }
    
    def _analyze_warmup_requirements(self, day: datetime, stocks: List[Dict]) -> Dict:
        """Analyze warmup bar requirements and track rejections."""
        if not stocks:
            return {
                'passed_warmup': [],
                'failed_warmup': [],
                'bar_count_stats': {},
                'warmup_required': 0
            }
        
        session_cfg = self.config.session
        
        # Regular session
        session_start_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')
        warmup = session_cfg.REGULAR_WARMUP_MINUTES
        min_bars = session_cfg.REGULAR_MIN_BARS
        
        session_end_et = pd.Timestamp(day.replace(hour=20, minute=0), tz='US/Eastern')
        session_start_utc = session_start_et.tz_convert('UTC')
        session_end_utc = session_end_et.tz_convert('UTC')
        
        passed_warmup = []
        failed_warmup = []
        bar_counts = []
        
        self.logger.info(f"  • Warmup Requirements:")
        self.logger.info(f"    - Warmup Period: {warmup} minutes")
        self.logger.info(f"    - Min Bars Required: {min_bars}")
        self.logger.info(f"    - Session Start: {session_start_et.strftime('%H:%M %Z')}")
        self.logger.info(f"  • Checking {len(stocks)} stocks for sufficient bars...")
        
        for stock in stocks:
            symbol = stock['symbol']
            
            # Get bars for the session
            bars = self.data_handler.get_intraday_bars(
                symbol,
                start=session_start_utc,
                end=session_end_utc
            )
            
            bar_count = len(bars) if not bars.empty else 0
            bar_counts.append(bar_count)
            
            # Check warmup requirements
            if bars.empty:
                failed_warmup.append({
                    'symbol': symbol,
                    'reason': 'empty_bars',
                    'bar_count': 0,
                    'gap_percent': stock['gap_percent']
                })
                continue
            
            if bar_count < warmup:
                failed_warmup.append({
                    'symbol': symbol,
                    'reason': 'insufficient_warmup_bars',
                    'bar_count': bar_count,
                    'warmup_required': warmup,
                    'gap_percent': stock['gap_percent']
                })
                continue
            
            # Passed warmup check
            stock['bar_count'] = bar_count
            passed_warmup.append(stock)
        
        self.logger.info(f"  • Warmup Check Results:")
        self.logger.info(f"    - Passed: {len(passed_warmup)} stocks")
        self.logger.info(f"    - Failed: {len(failed_warmup)} stocks")
        
        return {
            'passed_warmup': passed_warmup,
            'failed_warmup': failed_warmup,
            'bar_count_stats': {},
            'warmup_required': warmup,
            'min_bars': min_bars
        }
    
    def _analyze_patterns(self, day: datetime, stocks: List[Dict]) -> Dict:
        """Analyze pattern detection stage."""
        return {
            'valid_patterns': [],
            'failed_patterns': [],
            'failure_reasons': Counter()
        }
    
    def _analyze_risk_checks(self, day: datetime, valid_patterns: List[Dict]) -> Dict:
        """Analyze risk management checks."""
        return {
            'tradeable': [],
            'rejected': [],
            'rejection_reasons': Counter()
        }
    
    def _capture_config_snapshot(self):
        """Capture current configuration settings."""
        cfg = self.config
        
        self.results['config_snapshot'] = {
            # Screening config
            'screening_min_gap_percent': cfg.screening.MIN_GAP_PERCENT,
            'screening_min_price': cfg.screening.MIN_PRICE,
            'screening_max_price': cfg.screening.MAX_PRICE,
            'screening_min_cumulative_volume': cfg.screening.MIN_CUMULATIVE_VOLUME,
            'screening_min_daily_volume': cfg.screening.MIN_DAILY_VOLUME,
            'screening_enable_daily_volume_prescreen': cfg.screening.ENABLE_DAILY_VOLUME_PRESCREEN,
            'screening_min_relative_volume': cfg.screening.MIN_RELATIVE_VOLUME,
            'screening_enable_relative_volume': cfg.screening.ENABLE_RELATIVE_VOLUME,
            'screening_cumulative_volume_enabled': cfg.screening.CUMULATIVE_VOLUME,
            
            # Session config
            'session_premarket_enabled': cfg.session.PREMARKET_ENABLED,
            'session_premarket_warmup_minutes': cfg.session.PREMARKET_WARMUP_MINUTES,
            'session_premarket_min_bars': cfg.session.PREMARKET_MIN_BARS,
            'session_regular_warmup_minutes': cfg.session.REGULAR_WARMUP_MINUTES,
            'session_regular_min_bars': cfg.session.REGULAR_MIN_BARS,
            
            # Pattern config
            'pattern_min_step_ups': cfg.pattern.MIN_STEP_UPS,
            'pattern_min_advance_retention': cfg.pattern.MIN_ADVANCE_RETENTION,
            'pattern_max_pullback_percent': cfg.pattern.MAX_PULLBACK_PERCENT,
            'pattern_confluence_min_score': cfg.pattern.CONFLUENCE_NORMAL_GAP_MIN_SCORE,
            'pattern_confluence_min_patterns': cfg.pattern.CONFLUENCE_MIN_PATTERNS,
            
            # Risk config
            'risk_stop_loss_percent_of_account': cfg.risk.STOP_LOSS_PERCENT_OF_ACCOUNT,
            'risk_max_hold_time_minutes': cfg.risk.MAX_HOLD_TIME_MINUTES,
            'risk_max_position_size_percent': cfg.risk.MAX_POSITION_SIZE_PERCENT,
            'risk_max_daily_loss_percent': cfg.risk.MAX_DAILY_LOSS_PERCENT,
            'risk_max_concurrent_positions': cfg.risk.MAX_CONCURRENT_POSITIONS,
            'risk_atr_trailing_enabled': cfg.risk.ATR_TRAILING_ENABLED,
            'risk_atr_trailing_multiplier': cfg.risk.ATR_TRAILING_MULTIPLIER,
            
            # Backtest config
            'backtest_start_date': cfg.backtest.START_DATE,
            'backtest_end_date': cfg.backtest.END_DATE,
            'backtest_initial_capital': cfg.backtest.INITIAL_CAPITAL,
            'backtest_base_data_dir': cfg.backtest.BASE_DATA_DIR,
            'backtest_data_dir': cfg.backtest.DATA_DIR,
            'backtest_entry_cutoff_minutes': cfg.backtest.ENTRY_CUTOFF_MINUTES,
            'backtest_max_candidates_per_day': cfg.backtest.MAX_CANDIDATES_PER_DAY,
            
            # System config
            'system_reports_dir': cfg.system.REPORTS_DIR,
            'system_log_level': cfg.system.LOG_LEVEL,
            'system_max_day_cache_size': cfg.system.MAX_DAY_CACHE_SIZE,
            'system_use_file_index_cache': cfg.system.USE_FILE_INDEX_CACHE,
        }
    
    def _log_daily_summary(self, summary: Dict):
        """Log summary for a single day."""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"DAY SUMMARY: {summary['date']}")
        self.logger.info("=" * 80)
        
        stages = summary['stage']
        
        if 'universe' in stages:
            u = stages['universe']
            self.logger.info(f"Universe: {u['total_universe']:,} symbols | With Data: {u['with_data']}")
        
        if 'gaps' in stages:
            g = stages['gaps']
            self.logger.info(f"Gaps: {g['total_gaps']} calculated")
        
        if 'screening' in stages:
            s = stages['screening']
            passed = len(s.get('passed_stocks', []))
            self.logger.info(f"Screening: {passed} passed all filters")
        
        if 'warmup' in stages:
            w = stages['warmup']
            self.logger.info(f"Warmup: {len(w.get('passed_warmup', []))} passed | {len(w.get('failed_warmup', []))} failed")
        
        if 'patterns' in stages:
            p = stages['patterns']
            self.logger.info(f"Patterns: {len(p.get('valid_patterns', []))} valid | {len(p.get('failed_patterns', []))} failed")
        
        if 'risk' in stages:
            r = stages['risk']
            self.logger.info(f"Risk: {len(r.get('tradeable', []))} tradeable")
        
        self.logger.info("=" * 80)
    
    def _generate_comprehensive_report(self) -> pd.DataFrame:
        """Generate comprehensive report with actual CSV exports."""
        from pathlib import Path
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("GENERATING COMPREHENSIVE REPORT")
        self.logger.info("=" * 80)
        
        # Create reports directory
        reports_dir = Path(self.config.system.REPORTS_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamped diagnostic run directory
        from datetime import datetime
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        diagnostic_dir = reports_dir / f"diagnostic_{run_timestamp}"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory: {diagnostic_dir.absolute()}")
        
        # Initialize records lists
        daily_records = []
        gap_details = []
        screening_details = []
        warmup_details = []
        
        # 1. Export config snapshot
        if self.results['config_snapshot']:
            config_df = pd.DataFrame([self.results['config_snapshot']])
            config_file = diagnostic_dir / "config_snapshot.csv"
            config_df.to_csv(config_file, index=False)
            self.logger.info(f"✓ Config snapshot: {config_file.name}")
        
        # 2. Export daily summaries WITH detailed metrics
        if self.results['daily_summaries']:
            for summary in self.results['daily_summaries']:
                date = summary['date']
                stages = summary['stage']
                
                record = {'date': date}
                
                # Universe stats
                if 'universe' in stages:
                    u = stages['universe']
                    record['universe_total'] = u['total_universe']
                    record['universe_with_data'] = u['with_data']
                    record['universe_without_data'] = u['without_data']
                
                # Gap stats
                if 'gaps' in stages:
                    g = stages['gaps']
                    record['gaps_total'] = g['total_gaps']
                    record['gaps_prev_day'] = g.get('prev_day_used', 'N/A')
                    record['gaps_lookback_days'] = g.get('lookback_days', 0)
                    
                    if 'gap_distribution' in g:
                        record['gap_min'] = g['gap_distribution']['min']
                        record['gap_max'] = g['gap_distribution']['max']
                        record['gap_mean'] = g['gap_distribution']['mean']
                        record['gap_median'] = g['gap_distribution']['median']
                    
                    if 'gap_ranges' in g:
                        for range_name, count in g['gap_ranges'].items():
                            record[f'gap_range_{range_name}'] = count
                    
                    if 'top_gapper' in g:
                        record['top_gapper_symbol'] = g['top_gapper']['symbol']
                        record['top_gapper_pct'] = g['top_gapper']['gap_percent']
                        record['top_gapper_price'] = g['top_gapper']['last_price']
                
                # Screening stats
                if 'screening' in stages:
                    s = stages['screening']
                    fr = s.get('filter_results', {})
                    record['screen_initial'] = fr.get('initial', 0)
                    record['screen_after_gap'] = fr.get('after_gap_threshold', 0)
                    record['screen_after_price'] = fr.get('after_price_range', 0)
                    record['screen_after_volume'] = fr.get('after_volume', 0)
                    record['screen_after_rel_volume'] = fr.get('after_relative_volume', 0)
                    record['screen_passed'] = len(s.get('passed_stocks', []))
                
                # Warmup stats
                if 'warmup' in stages:
                    w = stages['warmup']
                    record['warmup_passed'] = len(w.get('passed_warmup', []))
                    record['warmup_failed'] = len(w.get('failed_warmup', []))
                    record['warmup_required_mins'] = w.get('warmup_required', 0)
                    record['warmup_min_bars'] = w.get('min_bars', 0)
                
                # Pattern stats
                if 'patterns' in stages:
                    p = stages['patterns']
                    record['patterns_valid'] = len(p.get('valid_patterns', []))
                    record['patterns_failed'] = len(p.get('failed_patterns', []))
                
                # Risk stats
                if 'risk' in stages:
                    r = stages['risk']
                    record['risk_tradeable'] = len(r.get('tradeable', []))
                    record['risk_rejected'] = len(r.get('rejected', []))
                
                daily_records.append(record)
                
                # Extract detailed gap data per stock per day
                if 'gaps' in stages and stages['gaps'].get('gaps_df') is not None:
                    gaps_df = stages['gaps']['gaps_df']
                    for _, gap_row in gaps_df.iterrows():
                        gap_details.append({
                            'date': date,
                            'symbol': gap_row['symbol'],
                            'gap_percent': gap_row['gap_percent'],
                            'open_price': gap_row['open_price'],
                            'prev_close': gap_row['prev_close'],
                            'last_price': gap_row['last_price'],
                            'volume': gap_row['volume']
                        })
                
                # Extract screening pass/fail details per stock
                if 'screening' in stages:
                    s = stages['screening']
                    for stock in s.get('passed_stocks', []):
                        screening_details.append({
                            'date': date,
                            'symbol': stock['symbol'],
                            'gap_percent': stock['gap_percent'],
                            'last_price': stock['last_price'],
                            'volume': stock['volume'],
                            'passed_screening': True,
                            'stage': 'screening_passed'
                        })
                
                # Extract warmup pass/fail details
                if 'warmup' in stages:
                    w = stages['warmup']
                    for stock in w.get('passed_warmup', []):
                        warmup_details.append({
                            'date': date,
                            'symbol': stock['symbol'],
                            'gap_percent': stock['gap_percent'],
                            'bar_count': stock.get('bar_count', 0),
                            'passed_warmup': True,
                            'reason': 'passed'
                        })
                    
                    for stock in w.get('failed_warmup', []):
                        warmup_details.append({
                            'date': date,
                            'symbol': stock['symbol'],
                            'gap_percent': stock['gap_percent'],
                            'bar_count': stock.get('bar_count', 0),
                            'passed_warmup': False,
                            'reason': stock.get('reason', 'unknown')
                        })
            
            # Export daily summary
            daily_df = pd.DataFrame(daily_records)
            daily_file = diagnostic_dir / "daily_summaries.csv"
            daily_df.to_csv(daily_file, index=False)
            self.logger.info(f"✓ Daily summaries: {daily_file.name} ({len(daily_df)} days)")
            
            # Export gap details
            if gap_details:
                gap_df = pd.DataFrame(gap_details)
                gap_file = diagnostic_dir / "gap_details.csv"
                gap_df.to_csv(gap_file, index=False)
                self.logger.info(f"✓ Gap details: {gap_file.name} ({len(gap_df)} records)")
            
            # Export screening details
            if screening_details:
                screen_df = pd.DataFrame(screening_details)
                screen_file = diagnostic_dir / "screening_details.csv"
                screen_df.to_csv(screen_file, index=False)
                self.logger.info(f"✓ Screening details: {screen_file.name} ({len(screen_df)} records)")
            
            # Export warmup details
            if warmup_details:
                warmup_df = pd.DataFrame(warmup_details)
                warmup_file = diagnostic_dir / "warmup_details.csv"
                warmup_df.to_csv(warmup_file, index=False)
                self.logger.info(f"✓ Warmup details: {warmup_file.name} ({len(warmup_df)} records)")
        
        # Export filter funnel analysis
        if daily_records:
            funnel_df = daily_df[[
                'date', 
                'screen_initial', 
                'screen_after_gap', 
                'screen_after_price', 
                'screen_after_volume',
                'warmup_passed',
                'warmup_failed',
                'patterns_valid',
                'risk_tradeable'
            ]].copy()
            
            # Add conversion rates
            if 'screen_initial' in funnel_df.columns:
                funnel_df['gap_filter_pass_rate'] = (funnel_df['screen_after_gap'] / funnel_df['screen_initial'] * 100).fillna(0)
                funnel_df['price_filter_pass_rate'] = (funnel_df['screen_after_price'] / funnel_df['screen_after_gap'] * 100).fillna(0)
                funnel_df['volume_filter_pass_rate'] = (funnel_df['screen_after_volume'] / funnel_df['screen_after_price'] * 100).fillna(0)
                funnel_df['warmup_pass_rate'] = (funnel_df['warmup_passed'] / (funnel_df['warmup_passed'] + funnel_df['warmup_failed']) * 100).fillna(0)
            
            funnel_file = diagnostic_dir / "filter_funnel_analysis.csv"
            funnel_df.to_csv(funnel_file, index=False)
            self.logger.info(f"✓ Filter funnel analysis: {funnel_file.name}")
        
        self.logger.info("=" * 80)
        self.logger.info(f"📁 All files saved to: {diagnostic_dir.absolute()}")
        self.logger.info("=" * 80)
        
        return pd.DataFrame(daily_records) if daily_records else pd.DataFrame()


def main():
    """Main entry point for diagnostic script."""
    
    # Force console logging
    import logging
    
    root_logger = logging.getLogger()
    
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream.name == '<stderr>'
        for h in root_logger.handlers
    )
    
    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.DEBUG)
        print("✓ Added console handler to root logger")
    
    # Configuration - January 2024 only
    START_DATE = "2024-01-02"  # First trading day of 2024
    END_DATE = "2024-01-31"     # Last day of January 2024
    
    print("=" * 100)
    print("COMPREHENSIVE TRADE DIAGNOSTIC - JANUARY 2024")
    print("=" * 100)
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Mode: Full universe analysis (all symbols)")
    print()
    
    try:
        print("[STEP 1/3] Loading configuration...")
        config = build_config()
        print("✓ Configuration loaded successfully")
        print(f"  - BASE_DATA_DIR: {config.backtest.BASE_DATA_DIR}")
        print(f"  - DATA_DIR: {config.backtest.DATA_DIR}")
        print(f"  - REPORTS_DIR: {config.system.REPORTS_DIR}")
        print(f"  - USE_FILE_INDEX_CACHE: {config.system.USE_FILE_INDEX_CACHE}")
        print()
        
    except Exception as e:
        print(f"✗ FAILED to load configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        print("[STEP 2/3] Initializing diagnostic tool...")
        diagnostic = ComprehensiveTradeDiagnostic(config)
        print("✓ Diagnostic tool initialized")
        print()
        
        # ✅ ADD DIAGNOSTIC: Check data handler file index
        print("[DIAGNOSTIC] Checking LocalDataHandler file index...")
        from pathlib import Path
        
        file_index = diagnostic.data_handler._file_index
        print(f"  - File index entries: {len(file_index)}")
        print(f"  - Total symbols across all dates: {sum(len(v) for v in file_index.values())}")
        
        # Show sample dates
        if file_index:
            print(f"\n  - Sample dates in file index (first 10):")
            for i, (date_str, symbols) in enumerate(sorted(file_index.items())[:10]):
                print(f"    {i+1}. {date_str}: {len(symbols)} symbols")
            
            # Check if January 2024 dates are present
            jan_2024_dates = [d for d in file_index.keys() if d.startswith('2024-01')]
            print(f"\n  - January 2024 dates in index: {len(jan_2024_dates)}")
            if jan_2024_dates:
                print(f"    Range: {min(jan_2024_dates)} to {max(jan_2024_dates)}")
                for date in sorted(jan_2024_dates)[:5]:
                    print(f"    • {date}: {len(file_index[date])} symbols")
        else:
            print("  ⚠️  File index is EMPTY!")
            
            # Run filesystem diagnostic
            print("\n[DIAGNOSTIC] Running filesystem diagnostic...")
            data_path = Path(config.backtest.DATA_DIR)
            print(f"  - Data directory: {data_path}")
            print(f"  - Directory exists: {data_path.exists()}")
            
            if data_path.exists():
                print("\n  - Checking directory structure...");
                
                # Check for year directories
                year_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 4]
                print(f"    Year directories found: {len(year_dirs)}")
                
                if year_dirs:
                    # Focus on 2024
                    year_2024 = data_path / "2024"
                    if year_2024.exists():
                        print(f"\n    Examining 2024/:")
                        month_dirs = [d for d in year_2024.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 2]
                        print(f"      • Month directories: {len(month_dirs)}")
                        
                        # Check January
                        jan_dir = year_2024 / "01"
                        if jan_dir.exists():
                            parquet_files = list(jan_dir.glob("*.parquet"))
                            print(f"\n      Examining 2024/01/:")
                            print(f"        • Parquet files: {len(parquet_files)}")
                            
                            if parquet_files:
                                # Test read one file
                                test_file = parquet_files[0]
                                print(f"\n        Testing file: {test_file.name}")
                                try:
                                    import pandas as pd
                                    df = pd.read_parquet(test_file)
                                    print(f"          ✓ Readable: {len(df)} rows")
                                    print(f"          ✓ Columns: {list(df.columns)}")
                                    if 'symbol' in df.columns:
                                        print(f"          ✓ Symbols: {len(df['symbol'].unique())}")
                                    else:
                                        print(f"          ✗ Missing 'symbol' column!")
                                except Exception as e:
                                    print(f"          ✗ Error reading: {e}")
                            else:
                                print(f"        ✗ No parquet files found in 2024/01/")
                        else:
                            print(f"      ✗ 2024/01/ directory doesn't exist")
                    else:
                        print(f"    ✗ 2024/ directory doesn't exist")
                else:
                    print(f"    ✗ No year directories found")
                    
                    # List what IS there
                    all_items = list(data_path.iterdir())
                    print(f"\n    Contents of {data_path}:")
                    for item in all_items[:20]:  # Show first 20 items
                        item_type = "DIR" if item.is_dir() else "FILE"
                        print(f"      [{item_type}] {item.name}")
            else:
                print(f"  ✗ Data directory does not exist: {data_path}")
            
            print()
            print("⚠️  STOPPING: Cannot proceed without valid file index")
            return
        
        print()
        
    except Exception as e:
        print(f"✗ FAILED to initialize diagnostic: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        print("[STEP 3/3] Running diagnostic analysis...")
        results = diagnostic.analyze_date_range(START_DATE, END_DATE)
        print("✓ Analysis complete")
        print(f"  - Results shape: {results.shape if not results.empty else 'Empty DataFrame'}")
        print()
        
    except Exception as e:
        print(f"✗ FAILED during analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)
    print(f"Processed {len(diagnostic.results['daily_summaries'])} trading days")
    print()


if __name__ == "__main__":
    main()