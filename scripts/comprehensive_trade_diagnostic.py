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

from config import TradingConfig
from data_handler.local import LocalDataHandler
from core.pattern_analyzer import PatternAnalyzer
from screener.backtest import BacktestScreener
from screener.rules import cfg_view_from
from utils.logging import get_logger


class ComprehensiveTradeDiagnostic:
    """Comprehensive diagnostic tool for analyzing the entire trading pipeline."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = get_logger(__name__, component="comprehensive_diagnostic")
        
        # Initialize components
        self.data_handler = LocalDataHandler(config)
        self.pattern_analyzer = PatternAnalyzer(config, self.data_handler)
        self.screener = BacktestScreener(config, self.data_handler, self.pattern_analyzer)
        
        # Results storage
        self.results = {
            'daily_summaries': [],
            'stock_details': [],
            'filter_stats': defaultdict(Counter),
            'pattern_failures': defaultdict(list),
            'warmup_rejections': [],
            'gap_conversion_losses': [],  # ✅ NEW: Track universe → gap losses      
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
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        self.logger.info("=" * 100)
        self.logger.info(f"COMPREHENSIVE TRADE DIAGNOSTIC")
        self.logger.info(f"Date Range: {start_date} to {end_date}")
        self.logger.info("=" * 100)
        
        # Capture configuration snapshot
        self._capture_config_snapshot()
        
        # Process each day
        current = start
        day_count = 0
        
        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                day_str = current.strftime('%Y-%m-%d')
                
                if self.data_handler.has_data_for_date(day_str):
                    self.logger.info("")
                    self.logger.info("=" * 100)
                    self.logger.info(f"ANALYZING: {day_str}")
                    self.logger.info("=" * 100)
                    
                    self._analyze_single_day(current)
                    day_count += 1
                else:
                    self.logger.info(f"SKIP: {day_str} - No data available")
            
            current += timedelta(days=1)
        
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
        gap_stats = self._analyze_gaps(day)
        daily_summary['stage']['gaps'] = gap_stats
        
        # STAGE 3: Screening Filters
        self.logger.info("[STAGE 3] SCREENING FILTERS")
        screening_stats = self._analyze_screening_filters(day, gap_stats.get('gaps_df'))
        daily_summary['stage']['screening'] = screening_stats
        
        # ✅ NEW: STAGE 3.5: Warmup Bar Analysis
        self.logger.info("[STAGE 3.5] WARMUP BAR ANALYSIS")
        warmup_stats = self._analyze_warmup_requirements(day, screening_stats.get('passed_stocks', []))
        daily_summary['stage']['warmup'] = warmup_stats
        
        # STAGE 4: Pattern Analysis (only stocks that passed warmup)
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
        
        # Check which symbols have data for this day
        symbols_with_data = []
        symbols_without_data = []
        
        for symbol in universe['symbol'].tolist()[:100]:  # Sample first 100 for performance
            day_data = self.data_handler.get_symbol_day_data(symbol, day_str)
            if not day_data.empty:
                symbols_with_data.append(symbol)
            else:
                symbols_without_data.append(symbol)
        
        stats = {
            'total_universe': total_symbols,
            'sampled': min(100, total_symbols),
            'with_data': len(symbols_with_data),
            'without_data': len(symbols_without_data),
            'sample_symbols': symbols_with_data[:10]
        }
        
        self.logger.info(f"  • Total Universe: {total_symbols:,} symbols")
        self.logger.info(f"  • Sampled: {stats['sampled']} symbols")
        self.logger.info(f"  • With Data: {stats['with_data']} | Without Data: {stats['without_data']}")
        
        return stats
    
    def _analyze_gaps(self, day: datetime) -> Dict:
        """Analyze gap calculation stage."""
        session_cfg = self.config.session
        premarket_enabled = session_cfg.PREMARKET_ENABLED
        
        # Calculate gaps
        gaps_info = self.data_handler.calculate_gaps(day, premarket=premarket_enabled)
        gaps_df = gaps_info.get('gaps', pd.DataFrame())
        
        stats = {
            'premarket_enabled': premarket_enabled,
            'prev_day_used': gaps_info.get('prev_day_used', 'N/A'),
            'lookback_days': gaps_info.get('lookback_days', 0),
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
            
            # Top gappers
            top_gaps = gaps_df.nlargest(10, 'gap_percent')[['symbol', 'gap_percent', 'last_price']]
            stats['top_gapper_max'] = float(top_gaps['gap_percent'].max())
            stats['top_gapper_avg'] = float(top_gaps['gap_percent'].mean())
            stats['top_gappers'] = top_gaps.to_dict('records')
        
        self.logger.info(f"  • Premarket Mode: {premarket_enabled}")
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
        
        cfg = cfg_view_from(self.config)
        
        # Track each filter stage
        filter_results = {
            'initial': len(gaps_df),
            'after_gap_threshold': 0,
            'after_price_range': 0,
            'after_volume': 0,
            'after_relative_volume': 0
        }
        
        # Filter 1: Gap Threshold
        min_gap = self.config.screening.MIN_GAP_PERCENT
        after_gap = gaps_df[gaps_df['gap_percent'].abs() >= min_gap]
        filter_results['after_gap_threshold'] = len(after_gap)
        
        self.logger.info(f"  • [FILTER 1] Gap Threshold (>= {min_gap}%)")
        self.logger.info(f"    Before: {filter_results['initial']} | After: {filter_results['after_gap_threshold']} | Filtered: {filter_results['initial'] - filter_results['after_gap_threshold']}")
        
        # Filter 2: Price Range
        after_price = after_gap[
            (after_gap['last_price'] >= cfg.MIN_PRICE) &
            (after_gap['last_price'] <= cfg.MAX_PRICE)
        ]
        filter_results['after_price_range'] = len(after_price)
        
        self.logger.info(f"  • [FILTER 2] Price Range (${cfg.MIN_PRICE:.2f} - ${cfg.MAX_PRICE:.2f})")
        self.logger.info(f"    Before: {filter_results['after_gap_threshold']} | After: {filter_results['after_price_range']} | Filtered: {filter_results['after_gap_threshold'] - filter_results['after_price_range']}")
        
        # Filter 3: Absolute Volume
        day_str = day.strftime('%Y-%m-%d')
        min_abs_vol = self.config.screening.MIN_ABSOLUTE_VOLUME
        
        stocks_with_volume = []
        for _, row in after_price.iterrows():
            symbol = row['symbol']
            bars = self.data_handler.get_symbol_day_data(symbol, day_str)
            if not bars.empty:
                total_volume = bars['volume'].sum()
                if total_volume >= min_abs_vol:
                    stocks_with_volume.append({
                        'symbol': symbol,
                        'gap_percent': row['gap_percent'],
                        'last_price': row['last_price'],
                        'volume': total_volume
                    })
        
        filter_results['after_volume'] = len(stocks_with_volume)
        
        self.logger.info(f"  • [FILTER 3] Absolute Volume (>= {min_abs_vol:,})")
        self.logger.info(f"    Before: {filter_results['after_price_range']} | After: {filter_results['after_volume']} | Filtered: {filter_results['after_price_range'] - filter_results['after_volume']}")
        
        # Filter 4: Relative Volume (if enabled)
        passed_stocks = stocks_with_volume
        
        if self.config.screening.ENABLE_RELATIVE_VOLUME:
            min_rel_vol = self.config.screening.MIN_RELATIVE_VOLUME
            
            stocks_with_rel_vol = []
            for stock in stocks_with_volume:
                symbol = stock['symbol']
                
                # Calculate relative volume
                end_date = day - timedelta(days=1)
                lookback = self.config.screening.RELATIVE_VOLUME_LOOKBACK_DAYS
                daily_bars = self.data_handler.get_daily_volume_history(
                    symbol, 
                    end_date - timedelta(days=lookback + 10),
                    end_date,
                    bars=lookback
                )
                
                if daily_bars is not None and not daily_bars.empty:
                    avg_volume = daily_bars['volume'].mean()
                    rel_vol = stock['volume'] / avg_volume if avg_volume > 0 else 0
                    
                    if rel_vol >= min_rel_vol:
                        stock['relative_volume'] = rel_vol
                        stocks_with_rel_vol.append(stock)
            
            filter_results['after_relative_volume'] = len(stocks_with_rel_vol)
            passed_stocks = stocks_with_rel_vol
            
            self.logger.info(f"  • [FILTER 4] Relative Volume (>= {min_rel_vol}x)")
            self.logger.info(f"    Before: {filter_results['after_volume']} | After: {filter_results['after_relative_volume']} | Filtered: {filter_results['after_volume'] - filter_results['after_relative_volume']}")
        else:
            filter_results['after_relative_volume'] = filter_results['after_volume']
            self.logger.info(f"  • [FILTER 4] Relative Volume - DISABLED")
        
        # Log top survivors
        if passed_stocks:
            self.logger.info(f"  • Top Stocks After Screening:")
            for stock in sorted(passed_stocks, key=lambda x: x['gap_percent'], reverse=True)[:5]:
                rel_vol_str = f" | RelVol: {stock.get('relative_volume', 'N/A'):.2f}x" if 'relative_volume' in stock else ""
                self.logger.info(f"    - {stock['symbol']}: {stock['gap_percent']:+.2f}% | ${stock['last_price']:.2f} | Vol: {stock['volume']:,}{rel_vol_str}")
        
        # Store detailed results for each stock
        for stock in passed_stocks:
            self.results['stock_details'].append({
                'date': day.strftime('%Y-%m-%d'),
                'symbol': stock['symbol'],
                'stage': 'screening_passed',
                'gap_percent': stock['gap_percent'],
                'price': stock['last_price'],
                'volume': stock['volume'],
                'relative_volume': stock.get('relative_volume')
            })
        
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
        premarket_enabled = session_cfg.PREMARKET_ENABLED
        
        # Define session boundaries
        if premarket_enabled:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            session_start_et = pd.Timestamp(day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern')
            warmup = session_cfg.PREMARKET_WARMUP_MINUTES
            min_bars = session_cfg.PREMARKET_MIN_BARS
        else:
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
                
                # Store warmup rejection
                self.results['warmup_rejections'].append({
                    'date': day.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'reason': 'empty_bars',
                    'bar_count': 0,
                    'warmup_required': warmup,
                    'gap_percent': stock['gap_percent'],
                    'price': stock['last_price']
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
                
                # Store warmup rejection
                self.results['warmup_rejections'].append({
                    'date': day.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'reason': 'insufficient_warmup_bars',
                    'bar_count': bar_count,
                    'warmup_required': warmup,
                    'gap_percent': stock['gap_percent'],
                    'price': stock['last_price']
                })
                
                continue
            
            # Passed warmup check
            stock['bar_count'] = bar_count
            passed_warmup.append(stock)
        
        # Calculate bar count statistics
        bar_count_stats = {}
        if bar_counts:
            bar_count_stats = {
                'min': min(bar_counts),
                'max': max(bar_counts),
                'mean': sum(bar_counts) / len(bar_counts),
                'median': sorted(bar_counts)[len(bar_counts) // 2]
            }
            
            # Bar count distribution
            bar_count_stats['distribution'] = {
                '0 bars': sum(1 for c in bar_counts if c == 0),
                '1-4 bars': sum(1 for c in bar_counts if 1 <= c < 5),
                '5-9 bars': sum(1 for c in bar_counts if 5 <= c < 10),
                '10-14 bars': sum(1 for c in bar_counts if 10 <= c < 15),
                '15-29 bars': sum(1 for c in bar_counts if 15 <= c < 30),
                '30+ bars': sum(1 for c in bar_counts if c >= 30)
            }
        
        # Log results
        self.logger.info(f"  • Warmup Check Results:")
        self.logger.info(f"    - Passed: {len(passed_warmup)} stocks")
        self.logger.info(f"    - Failed: {len(failed_warmup)} stocks")
        
        if bar_count_stats:
            self.logger.info(f"  • Bar Count Statistics:")
            self.logger.info(f"    - Range: {bar_count_stats['min']} to {bar_count_stats['max']} bars")
            self.logger.info(f"    - Mean: {bar_count_stats['mean']:.1f} bars")
            self.logger.info(f"    - Median: {bar_count_stats['median']} bars")
            self.logger.info(f"  • Bar Count Distribution:")
            for range_name, count in bar_count_stats['distribution'].items():
                self.logger.info(f"    - {range_name}: {count} stocks")
        
        # Log stocks that failed warmup
        if failed_warmup:
            self.logger.info(f"  • Warmup Failures (Top 10):")
            for failure in sorted(failed_warmup, key=lambda x: x['gap_percent'], reverse=True)[:10]:
                self.logger.info(
                    f"    - {failure['symbol']}: Gap {failure['gap_percent']:+.2f}% | "
                    f"Bars: {failure['bar_count']} / {warmup} required | "
                    f"Reason: {failure['reason']}"
                )
        
        return {
            'passed_warmup': passed_warmup,
            'failed_warmup': failed_warmup,
            'bar_count_stats': bar_count_stats,
            'warmup_required': warmup,
            'min_bars': min_bars
        }
    
    def _analyze_patterns(self, day: datetime, stocks: List[Dict]) -> Dict:
        """Analyze pattern detection stage (only for stocks that passed warmup)."""
        if not stocks:
            return {
                'valid_patterns': [],
                'failed_patterns': [],
                'failure_reasons': Counter()
            }
        
        session_cfg = self.config.session
        premarket_enabled = session_cfg.PREMARKET_ENABLED
        
        if premarket_enabled:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            session_start_et = pd.Timestamp(day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern')
            warmup = session_cfg.PREMARKET_WARMUP_MINUTES
        else:
            session_start_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')
            warmup = session_cfg.REGULAR_WARMUP_MINUTES
        
        session_end_et = pd.Timestamp(day.replace(hour=20, minute=0), tz='US/Eastern')
        session_start_utc = session_start_et.tz_convert('UTC')
        session_end_utc = session_end_et.tz_convert('UTC')
        
        valid_patterns = []
        failed_patterns = []
        failure_reasons = Counter()
        
        self.logger.info(f"  • Analyzing {len(stocks)} stocks for patterns...")
        self.logger.info(f"  • Pattern Requirements:")
        self.logger.info(f"    - Min Score: {self.config.pattern.CONFLUENCE_MIN_SCORE}")
        self.logger.info(f"    - Min Patterns: {self.config.pattern.CONFLUENCE_MIN_PATTERNS}")
        self.logger.info(f"    - Parabolic Vol Multiplier: {self.config.pattern.PARABOLIC_MIN_VOL_MULTIPLIER}x")
        self.logger.info(f"    - Breakout Vol Multiplier: {self.config.pattern.BREAKOUT_VOL_MULTIPLIER}x")
        
        for stock in stocks:
            symbol = stock['symbol']
            gap_pct = stock.get('gap_percent', 0.0)  # ✅ Extract gap percent
            
            bars = self.data_handler.get_intraday_bars(
                symbol,
                start=session_start_utc,
                end=session_end_utc
            )
            
            bars_warm = bars.iloc[:warmup].copy()
            pattern_result = self.pattern_analyzer.analyze_pattern(
                symbol,
                bars=bars_warm,
                is_premarket=premarket_enabled
            )
            
            # ✅ FIX: Use .get() with defaults to handle None/empty dicts
            step_up = pattern_result.get('step_ups') if pattern_result.get('step_ups') is not None else {}
            parabolic = pattern_result.get('parabolic') if pattern_result.get('parabolic') is not None else {}
            breakout = pattern_result.get('breakout') if pattern_result.get('breakout') is not None else {}
            volume = pattern_result.get('volume') if pattern_result.get('volume') is not None else {}
            sr = pattern_result.get('support_resistance') if pattern_result.get('support_resistance') is not None else {}
            
            if pattern_result.get('valid', False):
                valid_patterns.append({
                    'symbol': symbol,
                    'gap_percent': gap_pct,
                    'bar_count': stock.get('bar_count', len(bars)),
                    'pattern_strength': pattern_result.get('pattern_strength', 0),
                    'patterns_detected': pattern_result.get('patterns_detected', []),
                    'pattern_count': pattern_result.get('pattern_count', 0)
                })
                
                self.results['stock_details'].append({
                    'date': day.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'stage': 'pattern_valid',
                    'gap_percent': gap_pct,
                    'bar_count': stock.get('bar_count', len(bars)),
                    'pattern_strength': pattern_result.get('pattern_strength', 0),
                    'patterns_detected': '|'.join(pattern_result.get('patterns_detected', []))
                })
            else:
                # Detailed failure analysis
                failure_details = []
                
                if pattern_result.get('pattern_strength', 0) < self.config.pattern.CONFLUENCE_MIN_SCORE:
                    failure_details.append(f"low_score({pattern_result.get('pattern_strength', 0):.1f})")
                
                if not step_up.get('detected', False):
                    if step_up.get('step_count', 0) < self.config.pattern.MIN_STEP_UPS:
                        failure_details.append(f"step_ups({step_up.get('step_count', 0)})")
                    elif step_up.get('retention_rate', 0) < self.config.pattern.MIN_ADVANCE_RETENTION:
                        failure_details.append(f"retention({step_up.get('retention_rate', 0):.1f}%)")
                
                if not parabolic.get('detected', False):
                    if parabolic.get('angle', 0) < self.config.pattern.PARABOLIC_MIN_ANGLE:
                        failure_details.append(f"angle({parabolic.get('angle', 0):.1f}°)")
                    elif parabolic.get('acceleration', 0) < self.config.pattern.PARABOLIC_MIN_ACCELERATION:
                        failure_details.append(f"accel({parabolic.get('acceleration', 0):.4f})")
                    elif parabolic.get('volume_multiplier', 0) < self.config.pattern.PARABOLIC_MIN_VOL_MULTIPLIER:
                        failure_details.append(f"vol_mult({parabolic.get('volume_multiplier', 0):.1f}x)")
                
                if not breakout.get('detected', False):
                    if breakout.get('volume_ratio', 0) < self.config.pattern.BREAKOUT_VOL_MULTIPLIER:
                        failure_details.append(f"breakout_vol({breakout.get('volume_ratio', 0):.1f}x)")
                
                reason = 'pattern_invalid:' + ','.join(failure_details) if failure_details else 'pattern_invalid'
                failure_reasons[reason] += 1
                
                # ✅ FIX: Store ALL required fields including gap_percent
                self.results['pattern_failures'][symbol].append({
                    'date': day.strftime('%Y-%m-%d'),
                    'gap_percent': gap_pct,  # ✅ NOW INCLUDED
                    'reason': reason,
                    'bar_count': stock.get('bar_count', len(bars)),
                    'pattern_strength': pattern_result.get('pattern_strength', 0),
                    'pattern_count': pattern_result.get('pattern_count', 0),
                    'patterns_detected': '|'.join(pattern_result.get('patterns_detected', [])),
                    'details': {
                        'step_up': step_up,
                        'parabolic': parabolic,
                        'breakout': breakout,
                        'volume': volume,
                        'support_resistance': sr
                    }
                })
            
            # Log detailed pattern analysis
            self.logger.info(f"    [{symbol}] Detailed Breakdown:")
            self.logger.info(f"      Bars: {stock.get('bar_count', len(bars))} | Gap: {gap_pct:+.2f}%")
            self.logger.info(f"      Step-Up: detected={step_up.get('detected', False)} | "
                             f"count={step_up.get('step_count', 0)} | "
                             f"retention={step_up.get('retention_rate', 0):.1f}% | "
                             f"strength={step_up.get('strength', 0):.1f}")
            self.logger.info(f"      Parabolic: detected={parabolic.get('detected', False)} | "
                             f"angle={parabolic.get('angle', 0):.1f}° | "
                             f"accel={parabolic.get('acceleration', 0):.4f} | "
                             f"vol_mult={parabolic.get('volume_multiplier', 0):.2f}x | "
                             f"strength={parabolic.get('strength', 0):.1f}")
            self.logger.info(f"      Breakout: detected={breakout.get('detected', False)} | "
                             f"vol_ratio={breakout.get('volume_ratio', 0):.2f}x | "
                             f"strength={breakout.get('strength', 0):.1f}")
            self.logger.info(f"      Volume: trend={volume.get('volume_trend', 'unknown')} | strength={volume.get('strength', 0):.1f}")
            self.logger.info(f"      Support/Resistance: strength={sr.get('strength', 0):.1f}")
            self.logger.info(f"      TOTAL SCORE: {pattern_result.get('pattern_strength', 0):.1f} / {self.config.pattern.CONFLUENCE_MIN_SCORE}")
        
        self.logger.info(f"  • Pattern Analysis Results:")
        self.logger.info(f"    - Valid: {len(valid_patterns)}")
        self.logger.info(f"    - Failed: {len(failed_patterns)}")
        
        if failure_reasons:
            self.logger.info(f"  • Top Failure Reasons:")
            for reason, count in failure_reasons.most_common(5):
                self.logger.info(f"    - {reason}: {count}")
        
        if valid_patterns:
            self.logger.info(f"  • Valid Patterns:")
            for pattern in valid_patterns[:5]:
                self.logger.info(
                    f"    - {pattern['symbol']}: Score {pattern['pattern_strength']:.1f} | "
                    f"Bars: {pattern['bar_count']} | "
                    f"Patterns: {', '.join(pattern['patterns_detected'])}"
                )
        
        return {
            'valid_patterns': valid_patterns,
            'failed_patterns': failed_patterns,
            'failure_reasons': failure_reasons
        }
    
    def _analyze_risk_checks(self, day: datetime, valid_patterns: List[Dict]) -> Dict:
        """Analyze risk management and position sizing checks."""
        if not valid_patterns:
            return {
                'tradeable': [],
                'rejected': [],
                'rejection_reasons': Counter()
            }
        
        tradeable = []
        rejected = []
        rejection_reasons = Counter()
        
        self.logger.info(f"  • Checking {len(valid_patterns)} stocks against risk rules...")
        
        for pattern in valid_patterns:
            symbol = pattern['symbol']
            
            # These would require full backtester context, so we'll do simplified checks
            # In practice, you'd need entry price, stop price, etc.
            
            tradeable.append(pattern)
        
        self.logger.info(f"  • Risk Check Results:")
        self.logger.info(f"    - Tradeable: {len(tradeable)}")
        self.logger.info(f"    - Rejected: {len(rejected)}")
        
        return {
            'tradeable': tradeable,
            'rejected': rejected,
            'rejection_reasons': rejection_reasons
        }
    
    def _capture_config_snapshot(self):
        """Capture current configuration settings."""
        self.results['config_snapshot'] = {
            'screening': {
                'MIN_GAP_PERCENT': self.config.screening.MIN_GAP_PERCENT,
                'MIN_PRICE': self.config.screening.MIN_PRICE,
                'MAX_PRICE': self.config.screening.MAX_PRICE,
                'MIN_ABSOLUTE_VOLUME': self.config.screening.MIN_ABSOLUTE_VOLUME,
                'ENABLE_RELATIVE_VOLUME': self.config.screening.ENABLE_RELATIVE_VOLUME,
                'MIN_RELATIVE_VOLUME': self.config.screening.MIN_RELATIVE_VOLUME,
                'RELATIVE_VOLUME_LOOKBACK_DAYS': self.config.screening.RELATIVE_VOLUME_LOOKBACK_DAYS
            },
            'session': {
                'PREMARKET_ENABLED': self.config.session.PREMARKET_ENABLED,
                'PREMARKET_START_ET': self.config.session.PREMARKET_START_ET,
                'PREMARKET_WARMUP_MINUTES': self.config.session.PREMARKET_WARMUP_MINUTES,
                'PREMARKET_MIN_BARS': self.config.session.PREMARKET_MIN_BARS,
                'REGULAR_WARMUP_MINUTES': self.config.session.REGULAR_WARMUP_MINUTES,
                'REGULAR_MIN_BARS': self.config.session.REGULAR_MIN_BARS
            },
            'pattern': {
                'MIN_STEP_UPS': self.config.pattern.MIN_STEP_UPS,
                'MIN_ADVANCE_RETENTION': self.config.pattern.MIN_ADVANCE_RETENTION,
                'PARABOLIC_MIN_ANGLE': self.config.pattern.PARABOLIC_MIN_ANGLE,
                'PARABOLIC_MIN_ACCELERATION': self.config.pattern.PARABOLIC_MIN_ACCELERATION,
                'PARABOLIC_MIN_VOL_MULTIPLIER': self.config.pattern.PARABOLIC_MIN_VOL_MULTIPLIER,
                'BREAKOUT_VOL_MULTIPLIER': self.config.pattern.BREAKOUT_VOL_MULTIPLIER,
                'CONFLUENCE_MIN_SCORE': self.config.pattern.CONFLUENCE_MIN_SCORE,
                'CONFLUENCE_MIN_PATTERNS': self.config.pattern.CONFLUENCE_MIN_PATTERNS
            },
            'risk': {
                'MAX_RISK_PER_TRADE': self.config.risk.MAX_RISK_PER_TRADE,
                'MAX_CONCURRENT_POSITIONS': self.config.risk.MAX_CONCURRENT_POSITIONS,
                'PROFIT_TARGET_CENTS': self.config.risk.PROFIT_TARGET_CENTS
            }
        }
    
    def _log_daily_summary(self, summary: Dict):
        """Log summary for a single day."""
        self.logger.info("")
        self.logger.info("=" * 100)
        self.logger.info(f"DAY SUMMARY: {summary['date']}")
        self.logger.info("=" * 100)
        
        stages = summary['stage']
        
        # Universe
        if 'universe' in stages:
            u = stages['universe']
            self.logger.info(f"Universe: {u['total_universe']:,} symbols | Sampled: {u['sampled']} | With Data: {u['with_data']}")
        
        # Gaps
        if 'gaps' in stages:
            g = stages['gaps']
            self.logger.info(f"Gaps: {g['total_gaps']} calculated | Premarket: {g['premarket_enabled']}")
        
        # Screening
        if 'screening' in stages:
            s = stages['screening']
            fr = s.get('filter_results', {})
            self.logger.info(f"Screening: {fr.get('initial', 0)} → {fr.get('after_gap_threshold', 0)} → {fr.get('after_price_range', 0)} → {fr.get('after_volume', 0)} → {fr.get('after_relative_volume', 0)} stocks")
        
        # ✅ NEW: Warmup
        if 'warmup' in stages:
            w = stages['warmup']
            self.logger.info(f"Warmup: {len(w.get('passed_warmup', []))} passed | {len(w.get('failed_warmup', []))} failed (required: {w.get('warmup_required', 0)} bars)")
        
        # Patterns
        if 'patterns' in stages:
            p = stages['patterns']
            self.logger.info(f"Patterns: {len(p.get('valid_patterns', []))} valid | {len(p.get('failed_patterns', []))} failed")
        
        # Risk
        if 'risk' in stages:
            r = stages['risk']
            self.logger.info(f"Risk: {len(r.get('tradeable', []))} tradeable | {len(r.get('rejected', []))} rejected")
        
        self.logger.info("=" * 100)
    
    def _generate_comprehensive_report(self) -> pd.DataFrame:
        """Generate comprehensive report from analysis."""
        self.logger.info("")
        self.logger.info("=" * 100)
        self.logger.info("GENERATING COMPREHENSIVE REPORT")
        self.logger.info("=" * 100)
        
        # Export config snapshot
        config_df = pd.DataFrame([self.results['config_snapshot']])
        config_file = os.path.abspath("diagnostic_config_snapshot.csv")
        config_df.to_csv(config_file, index=False)
        self.logger.info(f"✓ Configuration snapshot exported: {config_file}")
        
        # Export daily summaries
        daily_file = os.path.abspath("diagnostic_daily_summaries.csv")
        daily_records = []
        
        for summary in self.results['daily_summaries']:
            date = summary['date']
            stages = summary['stage']
            
            record = {'date': date}
            
            # Flatten stages
            if 'universe' in stages:
                record.update({f"universe_{k}": v for k, v in stages['universe'].items() if k != 'sample_symbols'})
            
            if 'gaps' in stages:
                g = stages['gaps']
                record['gaps_total'] = g['total_gaps']
                record['gaps_premarket'] = g['premarket_enabled']
                if 'gap_distribution' in g:
                    record.update({f"gap_{k}": v for k, v in g['gap_distribution'].items()})
            
            if 'screening' in stages:
                fr = stages['screening'].get('filter_results', {})
                record.update({f"screen_{k}": v for k, v in fr.items()})
            
            # ✅ NEW: Warmup stats
            if 'warmup' in stages:
                w = stages['warmup']
                record['warmup_passed'] = len(w.get('passed_warmup', []))
                record['warmup_failed'] = len(w.get('failed_warmup', []))
                record['warmup_required'] = w.get('warmup_required', 0)
                if 'bar_count_stats' in w and w['bar_count_stats']:
                    record.update({f"bars_{k}": v for k, v in w['bar_count_stats'].items() if k != 'distribution'})
            
            if 'patterns' in stages:
                p = stages['patterns']
                record['patterns_valid'] = len(p.get('valid_patterns', []))
                record['patterns_failed'] = len(p.get('failed_patterns', []))
            
            if 'risk' in stages:
                r = stages['risk']
                record['risk_tradeable'] = len(r.get('tradeable', []))
                record['risk_rejected'] = len(r.get('rejected', []))
            
            daily_records.append(record)
        
        if daily_records:
            daily_df = pd.DataFrame(daily_records)
            daily_df.to_csv(daily_file, index=False)
            self.logger.info(f"✓ Daily summaries exported: {daily_file}")
        
        # Export stock-level details
        if self.results['stock_details']:
            stock_df = pd.DataFrame(self.results['stock_details'])
            stock_file = os.path.abspath("diagnostic_stock_details.csv")
            stock_df.to_csv(stock_file, index=False)
            self.logger.info(f"✓ Stock details exported: {stock_file}")
        
        # ✅ NEW: Export warmup rejections
        if self.results['warmup_rejections']:
            warmup_df = pd.DataFrame(self.results['warmup_rejections'])
            warmup_file = os.path.abspath("diagnostic_warmup_rejections.csv")
            warmup_df.to_csv(warmup_file, index=False)
            self.logger.info(f"✓ Warmup rejections exported: {warmup_file}")
            
            # Log warmup rejection summary
            total_warmup_rejects = len(self.results['warmup_rejections'])
            reason_counts = warmup_df['reason'].value_counts()
            self.logger.info(f"  Warmup Rejection Summary:")
            self.logger.info(f"    Total: {total_warmup_rejects}")
            for reason, count in reason_counts.items():
                self.logger.info(f"    - {reason}: {count}")
        
        # ✅ FIXED: Export pattern failures with proper error handling
        if self.results['pattern_failures']:
            failure_records = []
            for symbol, failures in self.results['pattern_failures'].items():
                for failure in failures:
                    # ✅ Base record with safe defaults
                    record = {
                        'symbol': symbol,
                        'date': failure.get('date', ''),
                        'gap_percent': failure.get('gap_percent', 0.0),  # ✅ NOW CAPTURED
                        'bar_count': failure.get('bar_count', 0),
                        'reason': failure.get('reason', 'unknown'),
                        'pattern_strength': failure.get('pattern_strength', 0.0),
                        'pattern_count': failure.get('pattern_count', 0),
                        'patterns_detected': failure.get('patterns_detected', '')
                    }
                    
                    # ✅ Safely extract nested details with proper null handling
                    details = failure.get('details', {})
                    
                    # Step-up details
                    step_up = details.get('step_up', {}) if details.get('step_up') is not None else {}
                    record.update({
                        'step_up_detected': step_up.get('detected', False),
                        'step_up_count': step_up.get('step_count', 0),
                        'step_up_retention': step_up.get('retention_rate', 0.0),
                        'step_up_total_advance': step_up.get('total_advance', 0.0),
                        'step_up_strength': step_up.get('strength', 0.0)
                    })
                    
                    # Parabolic details
                    parabolic = details.get('parabolic', {}) if details.get('parabolic') is not None else {}
                    record.update({
                        'parabolic_detected': parabolic.get('detected', False),
                        'parabolic_angle': parabolic.get('angle', 0.0),
                        'parabolic_accel': parabolic.get('acceleration', 0.0),
                        'parabolic_vol_mult': parabolic.get('volume_multiplier', 0.0),
                        'parabolic_strength': parabolic.get('strength', 0.0)
                    })
                    
                    # Breakout details
                    breakout = details.get('breakout', {}) if details.get('breakout') is not None else {}
                    record.update({
                        'breakout_detected': breakout.get('detected', False),
                        'breakout_level': breakout.get('breakout_level', 0.0),
                        'breakout_range_size': breakout.get('range_size', 0.0),
                        'breakout_vol_ratio': breakout.get('volume_ratio', 0.0),
                        'breakout_strength': breakout.get('strength', 0.0)
                    })
                    
                    # Volume pattern details
                    volume = details.get('volume', {}) if details.get('volume') is not None else {}
                    record.update({
                        'volume_trend': volume.get('volume_trend', 'unknown'),
                        'volume_avg': volume.get('avg_volume', 0.0),
                        'volume_recent': volume.get('recent_volume', 0.0),
                        'volume_high_correlation': volume.get('high_volume_correlation', 0.0),
                        'volume_strength': volume.get('strength', 0.0)
                    })
                    
                    # Support/Resistance details
                    sr = details.get('support_resistance', {}) if details.get('support_resistance') is not None else {}
                    record.update({
                        'sr_current_price': sr.get('current_price', 0.0),
                        'sr_support_levels': len(sr.get('support', [])),
                        'sr_resistance_levels': len(sr.get('resistance', [])),
                        'sr_strength': sr.get('strength', 0.0)
                    })
                    
                    failure_records.append(record)
            
            if failure_records:
                failure_df = pd.DataFrame(failure_records)
                failure_file = os.path.abspath("diagnostic_pattern_failures.csv")
                failure_df.to_csv(failure_file, index=False)
                self.logger.info(f"✓ Pattern failures exported: {failure_file}")
                self.logger.info(f"  Rows: {len(failure_df)} | Columns: {len(failure_df.columns)}")
                self.logger.info(f"  Sample columns: {', '.join(failure_df.columns.tolist()[:10])}...")
        
        self.logger.info("=" * 100)
        self.logger.info("DIAGNOSTIC COMPLETE")
        self.logger.info("=" * 100)
        
        # Return daily summaries as main output
        return pd.DataFrame(daily_records) if daily_records else pd.DataFrame()


def main():
    """Main entry point for diagnostic script."""
    
    # Configuration
    START_DATE = "2024-01-03"  # Adjust to your date range
    END_DATE = "2024-01-12"    # Adjust to your date range
    
    print("=" * 100)
    print("COMPREHENSIVE TRADE DIAGNOSTIC")
    print("=" * 100)
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print()
    
    # Load configuration
    config = TradingConfig()
    
    # Run diagnostic
    diagnostic = ComprehensiveTradeDiagnostic(config)
    results = diagnostic.analyze_date_range(START_DATE, END_DATE)
    
    print()
    print("=" * 100)
    print("DIAGNOSTIC FILES GENERATED:")
    print("  • diagnostic_config_snapshot.csv - Configuration settings used")
    print("  • diagnostic_daily_summaries.csv - Daily pipeline statistics")
    print("  • diagnostic_stock_details.csv - Stock-level screening/pattern results")
    print("  • diagnostic_warmup_rejections.csv - Stocks rejected due to insufficient bars")
    print("  • diagnostic_pattern_failures.csv - Detailed pattern failure analysis")
    print("=" * 100)


if __name__ == "__main__":
    main()
