# =====================================================
# comprehensive_trade_diagnostic.py - In-Depth Backtest Diagnostic Tool
# =====================================================
"""
Comprehensive diagnostic script that analyzes the entire screening and pattern
analysis pipeline to understand why stocks that previously traded are now filtered out.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.loader import build_config
from data_handler.local import LocalDataHandler
from data_handler.aggregate_handler import AggregateDataHandler
from strategy.pattern_analyzer import PatternAnalyzer
from screener.helpers import BacktestRelativeVolumeCalculator
from market_context.backtest import BacktestMarketContext
from utils.logging import get_logger


class ComprehensiveTradeDiagnostic:
    """Comprehensive diagnostic tool for analyzing the entire trading pipeline."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, component="comprehensive_diagnostic")

        self.data_handler = LocalDataHandler(
            config,
            data_dir=config.backtest.DATA_DIR
        )

        self.pattern_analyzer = PatternAnalyzer(config, self.data_handler)

        agg_dir = Path(config.backtest.BASE_DATA_DIR) / "daily_aggregates"
        self._agg_handler = AggregateDataHandler(str(agg_dir))

        self._rvol_calculator = BacktestRelativeVolumeCalculator(config, self.data_handler, self.logger)

        try:
            self.market_context = BacktestMarketContext(config)
        except Exception as e:
            self.logger.warning(f"Market context init failed ({e}) — Filter 6 will be skipped")
            self.market_context = None

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
        print("\n" + "=" * 100)
        print("COMPREHENSIVE TRADE DIAGNOSTIC")
        print("=" * 100)

        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            print(f"✓ Parsed dates: {start.date()} to {end.date()}")
        except Exception as e:
            print(f"✗ Failed to parse dates: {e}")
            raise

        self.logger.info("=" * 100)
        self.logger.info("COMPREHENSIVE TRADE DIAGNOSTIC")
        self.logger.info(f"Date Range: {start_date} to {end_date}")
        self.logger.info("=" * 100)

        self._capture_config_snapshot()

        current = start
        days_processed = 0

        while current <= end:
            if current.weekday() < 5:
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

        return self._generate_comprehensive_report()

    def _analyze_single_day(self, day: datetime):
        day_str = day.strftime('%Y-%m-%d')
        daily_summary = {'date': day_str, 'stage': {}}

        self.logger.info("[STAGE 1] UNIVERSE DETECTION")
        daily_summary['stage']['universe'] = self._analyze_universe(day)

        self.logger.info("[STAGE 2] GAP CALCULATION")
        gap_stats = self._analyze_gaps_new(day)
        daily_summary['stage']['gaps'] = gap_stats

        self.logger.info("[STAGE 3] SCREENING FILTERS")
        screening_stats = self._analyze_screening_filters(day, gap_stats.get('gaps_df'))
        daily_summary['stage']['screening'] = screening_stats

        self.logger.info("[STAGE 3.5] WARMUP BAR ANALYSIS")
        warmup_stats = self._analyze_warmup_requirements(day, screening_stats.get('passed_stocks', []))
        daily_summary['stage']['warmup'] = warmup_stats

        self.logger.info("[STAGE 4] PATTERN ANALYSIS")
        pattern_stats = self._analyze_patterns(day, warmup_stats.get('passed_warmup', []))
        daily_summary['stage']['patterns'] = pattern_stats

        self.logger.info("[STAGE 5] POSITION SIZING & RISK")
        risk_stats = self._analyze_risk_checks(day, pattern_stats.get('valid_patterns', []))
        daily_summary['stage']['risk'] = risk_stats

        self.results['daily_summaries'].append(daily_summary)
        self._log_daily_summary(daily_summary)

    def _analyze_universe(self, day: datetime) -> Dict:
        day_str = day.strftime('%Y-%m-%d')
        universe = self.data_handler.get_universe()
        total_symbols = len(universe)

        sample_size = min(100, total_symbols)
        day_df = self.data_handler._get_day_df(day_str)
        day_symbols = (
            set(day_df['symbol'].unique())
            if not day_df.empty and 'symbol' in day_df.columns
            else set()
        )

        sampled = universe['symbol'].tolist()[:sample_size]
        with_data = sum(1 for s in sampled if s in day_symbols)

        stats = {
            'total_universe': total_symbols,
            'sampled': sample_size,
            'with_data': with_data,
            'without_data': sample_size - with_data,
            'sample_symbols': [s for s in sampled if s in day_symbols][:10]
        }

        self.logger.info(f"  • Total Universe: {total_symbols:,} | Sampled: {sample_size} | With Data: {with_data}")
        return stats

    def _analyze_gaps_new(self, day: datetime) -> Dict:
        """Gap calculation — mirrors screener/backtest.py exactly."""
        agg_handler = self._agg_handler
        day_agg = agg_handler.get_day_aggregates(day)

        if day_agg is None or day_agg.empty:
            self.logger.warning(f"No aggregate data for {day.strftime('%Y-%m-%d')}")
            return {'total_gaps': 0, 'gaps_df': pd.DataFrame(), 'prev_day_used': 'N/A', 'lookback_days': 0}

        prev_day = day - timedelta(days=1)
        prev_agg = None
        for lookback in range(1, 8):
            prev_agg = agg_handler.get_day_aggregates(prev_day)
            if prev_agg is not None and not prev_agg.empty:
                break
            prev_day -= timedelta(days=1)
        else:
            lookback = 7

        if prev_agg is None or prev_agg.empty:
            self.logger.warning("No previous day data found within 7 days")
            return {'total_gaps': 0, 'gaps_df': pd.DataFrame(), 'prev_day_used': 'N/A', 'lookback_days': 7}

        common_symbols = set(day_agg['symbol']) & set(prev_agg['symbol'])
        gaps_list = []
        for symbol in common_symbols:
            day_row = day_agg[day_agg['symbol'] == symbol].iloc[0]
            prev_row = prev_agg[prev_agg['symbol'] == symbol].iloc[0]
            prev_close = prev_row['close']
            if pd.isna(prev_close) or prev_close <= 0:
                continue
            day_open = day_row['open']
            if pd.isna(day_open) or day_open <= 0:
                continue
            gap_pct = ((day_open - prev_close) / prev_close) * 100.0
            if not np.isfinite(gap_pct):
                continue
            gaps_list.append({
                'symbol': symbol,
                'open_price': day_open,
                'prev_close': prev_close,
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
            g = gaps_df['gap_percent']
            stats['gap_distribution'] = {
                'min': float(g.min()), 'max': float(g.max()),
                'mean': float(g.mean()), 'median': float(g.median())
            }
            pos = gaps_df[gaps_df['gap_percent'] >= 0]['gap_percent']
            stats['gap_ranges'] = {
                'negative': int((g < 0).sum()),
                '0-5%':   int(((pos >= 0)  & (pos < 5)).sum()),
                '5-10%':  int(((pos >= 5)  & (pos < 10)).sum()),
                '10-20%': int(((pos >= 10) & (pos < 20)).sum()),
                '20-50%': int(((pos >= 20) & (pos < 50)).sum()),
                '50%+':   int((pos >= 50).sum()),
            }
            top_gap = gaps_df.loc[gaps_df['gap_percent'].idxmax()]
            stats['top_gapper'] = {
                'symbol': top_gap['symbol'],
                'gap_percent': float(top_gap['gap_percent']),
                'last_price': float(top_gap['last_price'])
            }
        else:
            stats['gap_distribution'] = {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
            stats['gap_ranges'] = {}

        self.logger.info(f"  • Previous Day: {stats['prev_day_used']} ({stats['lookback_days']} days back)")
        self.logger.info(f"  • Total Gaps: {stats['total_gaps']} | Breakdown: {stats['gap_ranges']}")
        return stats

    def _analyze_screening_filters(self, day: datetime, gaps_df: Optional[pd.DataFrame]) -> Dict:
        """Screening filters — mirrors UnifiedScreener.screen_symbols() gate order."""
        if gaps_df is None or gaps_df.empty:
            return {'passed_stocks': [], 'filter_results': {}, 'message': 'No gaps to analyze'}

        cfg_screening = self.config.screening
        filter_results = {
            'initial': len(gaps_df),
            'after_gap_threshold': 0,
            'after_price_range': 0,
            'after_volume': 0,
            'after_float_marketcap': 0,
            'after_rvol': 0,
            'market_context_blocked': False
        }

        # Filter 1: Gap Threshold — positive only
        min_gap = cfg_screening.MIN_GAP_PERCENT
        after_gap = gaps_df[gaps_df['gap_percent'] >= min_gap].copy()
        filter_results['after_gap_threshold'] = len(after_gap)
        self.logger.info(f"  • [F1] Gap >= {min_gap}%: {filter_results['initial']} → {filter_results['after_gap_threshold']}")

        # Filter 2: Price floor on aggregate close
        after_price = after_gap[after_gap['last_price'] >= cfg_screening.MIN_PRICE]
        filter_results['after_price_range'] = len(after_price)
        self.logger.info(f"  • [F2] Price >= ${cfg_screening.MIN_PRICE:.2f}: {filter_results['after_gap_threshold']} → {filter_results['after_price_range']}")

        # Filter 3: Daily Volume
        min_daily_vol = cfg_screening.MIN_DAILY_VOLUME
        after_volume = (
            after_price[after_price['volume'] >= min_daily_vol]
            if cfg_screening.ENABLE_DAILY_VOLUME_PRESCREEN
            else after_price.copy()
        )
        filter_results['after_volume'] = len(after_volume)
        self.logger.info(f"  • [F3] Daily Volume >= {min_daily_vol:,} (enabled={cfg_screening.ENABLE_DAILY_VOLUME_PRESCREEN}): {filter_results['after_price_range']} → {filter_results['after_volume']}")

        # Filter 4: Relative Volume (matches production gate order)
        after_rvol = after_volume.copy()
        if cfg_screening.ENABLE_RELATIVE_VOLUME:
            min_rvol = cfg_screening.MIN_RELATIVE_VOLUME
            try:
                symbols = after_volume['symbol'].tolist()
                rel_vols = self._rvol_calculator.calculate_batch(symbols, day)
                if rel_vols:
                    after_rvol = after_volume.copy()
                    after_rvol['relative_volume'] = after_rvol['symbol'].map(rel_vols).fillna(0.0)
                    after_rvol = after_rvol[after_rvol['relative_volume'] >= min_rvol]
                else:
                    self.logger.warning("  • [F4] RVOL enabled but calculator returned no data — filter skipped")
            except Exception as e:
                self.logger.warning(f"  • [F4] RVOL calculation error: {e} — filter skipped")
            self.logger.info(f"  • [F4] Relative Volume >= {min_rvol}x: {filter_results['after_volume']} → {len(after_rvol)}")
        else:
            self.logger.info("  • [F4] Relative Volume (disabled)")
        filter_results['after_rvol'] = len(after_rvol)

        # Filter 5: Float & Marketcap
        after_fundamentals = after_rvol.copy()
        if cfg_screening.ENABLE_FLOAT_FILTER or cfg_screening.ENABLE_MARKETCAP_FILTER:
            try:
                day_agg = self._agg_handler.get_day_aggregates(day)
                if day_agg is not None and not day_agg.empty:
                    agg_lookup = day_agg.set_index('symbol')

                    def passes_fundamentals(symbol):
                        if symbol not in agg_lookup.index:
                            return True
                        row = agg_lookup.loc[symbol]
                        if cfg_screening.ENABLE_FLOAT_FILTER and 'float' in row:
                            f = row['float']
                            if pd.notna(f) and (f < cfg_screening.MIN_FLOAT or f > cfg_screening.MAX_FLOAT):
                                return False
                        if cfg_screening.ENABLE_MARKETCAP_FILTER and 'marketcap' in row:
                            mc = row['marketcap']
                            if pd.notna(mc) and mc > cfg_screening.MAX_MARKETCAP:
                                return False
                        return True

                    after_fundamentals = after_rvol[after_rvol['symbol'].apply(passes_fundamentals)]
            except Exception as e:
                self.logger.warning(f"  • Fundamental filter error: {e} — skipping")
        filter_results['after_float_marketcap'] = len(after_fundamentals)
        self.logger.info(f"  • [F5] Float/Marketcap (enabled={cfg_screening.ENABLE_FLOAT_FILTER or cfg_screening.ENABLE_MARKETCAP_FILTER}): {filter_results['after_rvol']} → {filter_results['after_float_marketcap']}")

        # Filter 6: Market Context gate
        if self.market_context is not None:
            try:
                self.market_context.update_market_context(day)
                score = self.market_context.market_indicators.get('market_score', 0)
                env = self.market_context.market_indicators.get('trading_environment', 'unknown')
                if not self.market_context.should_trade():
                    self.logger.warning(f"  • [F6] Market Context BLOCKED — score={score:.1f}, env={env}")
                    filter_results['market_context_blocked'] = True
                    return {'passed_stocks': [], 'filter_results': filter_results}
                self.logger.info(f"  • [F6] Market Context OK — score={score:.1f}, env={env}")
            except Exception as e:
                self.logger.warning(f"  • [F6] Market Context error: {e} — not gating")
        else:
            self.logger.info("  • [F6] Market Context (not available — skipped)")

        passed_stocks = after_fundamentals.to_dict('records')
        if passed_stocks:
            top2 = sorted(passed_stocks, key=lambda x: x['gap_percent'], reverse=True)[:2]
            self.logger.info(f"  • {len(passed_stocks)} passed. Top 2: " +
                             " | ".join(f"{s['symbol']} {s['gap_percent']:+.1f}%" for s in top2))

        return {'passed_stocks': passed_stocks, 'filter_results': filter_results}

    def _analyze_warmup_requirements(self, day: datetime, stocks: List[Dict]) -> Dict:
        """Warmup + entry checks — mirrors UnifiedScreener.screen_symbols() per-symbol loop."""
        if not stocks:
            return {'passed_warmup': [], 'failed_warmup': [], 'bar_count_stats': {}, 'warmup_required': 0}

        session_cfg = self.config.session

        if session_cfg.PREMARKET_ENABLED:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            session_start_et = pd.Timestamp(
                day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern'
            )
            warmup = session_cfg.PREMARKET_WARMUP_MINUTES
            min_bars = session_cfg.PREMARKET_MIN_BARS
        else:
            session_start_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')
            warmup = session_cfg.REGULAR_WARMUP_MINUTES
            min_bars = session_cfg.REGULAR_MIN_BARS

        after_hours_end = datetime.strptime(session_cfg.AFTER_HOURS_END_ET, "%H:%M").time()
        session_end_et = pd.Timestamp(
            day.replace(hour=after_hours_end.hour, minute=after_hours_end.minute), tz='US/Eastern'
        )
        session_start_utc = session_start_et.tz_convert('UTC')
        session_end_utc = session_end_et.tz_convert('UTC')

        # Analysis window gate
        analysis_window_enabled = getattr(self.config.backtest, 'ANALYSIS_WINDOW_ENABLED', False)
        analysis_window_start_utc = None
        analysis_window_end_utc = None
        if analysis_window_enabled:
            w_start_str = getattr(self.config.backtest, 'ANALYSIS_WINDOW_START_ET', '06:00')
            w_end_str = getattr(self.config.backtest, 'ANALYSIS_WINDOW_END_ET', '12:00')
            ws = datetime.strptime(w_start_str, "%H:%M").time()
            we = datetime.strptime(w_end_str, "%H:%M").time()
            analysis_window_start_utc = pd.Timestamp(
                day.replace(hour=ws.hour, minute=ws.minute), tz='US/Eastern'
            ).tz_convert('UTC')
            analysis_window_end_utc = pd.Timestamp(
                day.replace(hour=we.hour, minute=we.minute), tz='US/Eastern'
            ).tz_convert('UTC')
            self.logger.info(f"  • Analysis Window: {w_start_str}–{w_end_str} ET (enabled)")
        else:
            self.logger.info("  • Analysis Window: disabled")

        self.logger.info(f"  • Session Start: {session_start_et.strftime('%H:%M %Z')} | Warmup: {warmup} bars | Min bars: {min_bars}")
        self.logger.info(f"  • Checking {len(stocks)} stocks...")

        passed_warmup = []
        failed_warmup = []
        bar_counts = []

        for stock in stocks:
            symbol = stock['symbol']
            bars = self.data_handler.get_intraday_bars(symbol, start=session_start_utc, end=session_end_utc)
            bar_count = len(bars) if not bars.empty else 0
            bar_counts.append(bar_count)

            if bars.empty:
                failed_warmup.append({'symbol': symbol, 'reason': 'empty_bars', 'bar_count': 0, 'gap_percent': stock['gap_percent']})
                continue

            if bar_count < warmup:
                failed_warmup.append({'symbol': symbol, 'reason': 'insufficient_warmup_bars', 'bar_count': bar_count, 'warmup_required': warmup, 'gap_percent': stock['gap_percent']})
                continue

            # NaN check before next-bar check — matches production gate order
            if bars.iloc[:warmup].isna().any().any():
                failed_warmup.append({'symbol': symbol, 'reason': 'nan_in_warmup_bars',
                                      'bar_count': bar_count, 'gap_percent': stock['gap_percent']})
                continue

            if bar_count <= warmup:
                failed_warmup.append({
                    'symbol': symbol, 'reason': 'no_next_bar_for_entry',
                    'bar_count': bar_count, 'warmup_required': warmup,
                    'gap_percent': stock['gap_percent']
                })
                continue

            bars_warm   = bars.iloc[:warmup]
            signal_bar  = bars.iloc[warmup - 1]   # last bar pattern runs on
            next_bar    = bars.iloc[warmup]        # first bar available for fill

            entry_price = float(next_bar['open'])          # ← fixed: was signal_bar['close']
            entry_ts_raw = next_bar.get('timestamp', None) # ← fixed: timestamp of fill bar

            if entry_price < self.config.screening.MIN_PRICE:
                failed_warmup.append({'symbol': symbol, 'reason': 'entry_price_below_min',
                                      'entry_price': entry_price, 'bar_count': bar_count,
                                      'gap_percent': stock['gap_percent']})
                continue

            # Analysis window check still uses fill bar's timestamp
            if analysis_window_start_utc is not None:
                if entry_ts_raw is not None:
                    entry_ts = pd.Timestamp(entry_ts_raw)
                    entry_ts = entry_ts.tz_localize('UTC') if entry_ts.tz is None else entry_ts.tz_convert('UTC')
                    if entry_ts < analysis_window_start_utc or entry_ts >= analysis_window_end_utc:
                        failed_warmup.append({'symbol': symbol, 'reason': 'outside_analysis_window',
                                              'bar_count': bar_count, 'gap_percent': stock['gap_percent']})
                        continue

            stock = stock.copy()
            stock['bar_count']        = bar_count
            stock['bars']             = bars
            stock['bars_warm']        = bars_warm.copy()
            stock['warmup']           = warmup
            stock['entry_price']      = entry_price                        # next_bar open
            stock['signal_bar_close'] = float(signal_bar['close'])         # audit: old behaviour
            stock['entry_ts']         = pd.Timestamp(entry_ts_raw) if entry_ts_raw is not None else None
            passed_warmup.append(stock)

        fail_summary = Counter(f['reason'] for f in failed_warmup)
        self.logger.info(f"  • Warmup: {len(passed_warmup)} passed | {len(failed_warmup)} failed — {dict(fail_summary)}")

        return {
            'passed_warmup': passed_warmup,
            'failed_warmup': failed_warmup,
            'bar_count_stats': {
                'mean': float(np.mean(bar_counts)) if bar_counts else 0,
                'min': int(min(bar_counts)) if bar_counts else 0,
                'max': int(max(bar_counts)) if bar_counts else 0
            },
            'warmup_required': warmup,
            'min_bars': min_bars
        }

    def _analyze_patterns(self, day: datetime, stocks: List[Dict]) -> Dict:
        valid_patterns = []
        failed_patterns = []
        failure_reasons = Counter()

        if not stocks:
            self.logger.info("  • No stocks passed warmup — skipping pattern analysis")
            return {'valid_patterns': valid_patterns, 'failed_patterns': failed_patterns, 'failure_reasons': failure_reasons}

        is_premarket = self.config.session.PREMARKET_ENABLED
        self.logger.info(f"  • Running pattern analysis on {len(stocks)} stocks...")

        for stock in stocks:
            symbol = stock['symbol']
            gap_percent = stock.get('gap_percent', None)
            try:
                bars_warm = stock.get('bars_warm')
                if bars_warm is None or bars_warm.empty:
                    reason = 'no_bars_warm'
                    failed_patterns.append({**stock, 'reason': reason})
                    failure_reasons[reason] += 1
                    continue

                result = self.pattern_analyzer.analyze_pattern(
                    symbol=symbol,
                    bars=bars_warm,
                    is_premarket=is_premarket,
                    gap_percent=gap_percent
                )

                if result.get('valid'):
                    valid_patterns.append({
                        **stock,
                        'pattern_strength': result.get('pattern_strength'),
                        'patterns_detected': result.get('patterns_detected'),  # FIX: was mismatched quote
                        'min_score_threshold': result.get('min_score_threshold')
                    })
                else:
                    reason = result.get('reason', 'pattern_invalid')
                    failed_patterns.append({
                        **stock,
                        'reason': reason,
                        'pattern_strength': result.get('pattern_strength', 0),
                        'min_score_threshold': result.get('min_score_threshold')
                    })
                    failure_reasons[reason] += 1

            except Exception as e:
                self.logger.error(f"  • Pattern error for {symbol}: {e}")
                reason = f'exception:{type(e).__name__}'
                failed_patterns.append({**stock, 'reason': reason})
                failure_reasons[reason] += 1

        self.logger.info(f"  • Pattern Results: {len(valid_patterns)} valid | {len(failed_patterns)} failed")
        self.logger.info(f"  • Top failure reasons: {dict(failure_reasons.most_common(5))}")
        return {'valid_patterns': valid_patterns, 'failed_patterns': failed_patterns, 'failure_reasons': failure_reasons}

    def _analyze_risk_checks(self, day: datetime, valid_patterns: List[Dict]) -> Dict:
        """Risk checks — mirrors backtest position sizing limits with per-reason codes."""
        tradeable = []
        rejected = []
        rejection_reasons = Counter()

        if not valid_patterns:
            self.logger.info("  • No valid patterns — skipping risk checks")
            return {'tradeable': tradeable, 'rejected': rejected, 'rejection_reasons': rejection_reasons}

        risk_cfg = self.config.risk
        backtest_cfg = self.config.backtest
        max_candidates = backtest_cfg.MAX_CANDIDATES_PER_DAY

        # ── Cap by MAX_CANDIDATES_PER_DAY (pre-sort by pattern strength) ────────
        sorted_patterns = sorted(valid_patterns, key=lambda x: x.get('pattern_strength', 0) or 0, reverse=True)
        capped_out = sorted_patterns[max_candidates:]
        after_cap = sorted_patterns[:max_candidates]
        if capped_out:
            rejection_reasons['exceeded_max_candidates_per_day'] += len(capped_out)
            rejected.extend([{**p, 'risk_rejected_reason': 'exceeded_max_candidates_per_day'} for p in capped_out])

        # ── Entry cutoff timestamp ───────────────────────────────────────────────
        session_cfg = self.config.session
        if session_cfg.PREMARKET_ENABLED:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            session_start_et = pd.Timestamp(
                day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern'
            )
        else:
            session_start_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')
        session_start_utc = session_start_et.tz_convert('UTC')
        # ── Entry cutoff: sourced from ANALYSIS_WINDOW_END_ET (single source of truth) ──
        w_end_str = getattr(self.config.backtest, 'ANALYSIS_WINDOW_END_ET', '16:00')
        we = datetime.strptime(w_end_str, "%H:%M").time()
        cutoff_utc = pd.Timestamp(
            day.replace(hour=we.hour, minute=we.minute), tz='US/Eastern'
        ).tz_convert('UTC')

        # ── Position sizing floor: can we afford even 1 share? ──────────────────
        initial_capital = backtest_cfg.INITIAL_CAPITAL
        max_pos_value = initial_capital * (risk_cfg.MAX_POSITION_SIZE_PERCENT / 100.0)
        risk_per_trade = initial_capital * (risk_cfg.STOP_LOSS_PERCENT_OF_ACCOUNT / 100.0)

        # ── Simulate daily-loss accumulator ─────────────────────────────────────
        daily_loss_limit = initial_capital * (risk_cfg.MAX_DAILY_LOSS_PERCENT / 100.0)
        simulated_daily_loss = 0.0   # conservative: assume no prior losses today

        concurrent_count = 0

        for pattern in after_cap:
            symbol = pattern['symbol']
            entry_price = pattern.get('entry_price', 0) or 0

            # ── Check 1: entry cutoff ────────────────────────────────────────────
            entry_ts = pattern.get('entry_ts')
            if entry_ts is not None:
                ts = pd.Timestamp(entry_ts)
                if ts.tz is None:
                    ts = ts.tz_localize('UTC')
                else:
                    ts = ts.tz_convert('UTC')
                if ts >= cutoff_utc:
                    reason = 'past_entry_cutoff'
                    rejection_reasons[reason] += 1
                    rejected.append({**pattern, 'risk_rejected_reason': reason,
                                     'risk_detail': f"entry_ts={ts.tz_convert('US/Eastern').strftime('%H:%M')} >= cutoff={cutoff_utc.tz_convert('US/Eastern').strftime('%H:%M')}"})
                    continue

            # ── Check 2: max concurrent positions ───────────────────────────────
            if concurrent_count >= risk_cfg.MAX_CONCURRENT_POSITIONS:
                reason = 'max_concurrent_positions'
                rejection_reasons[reason] += 1
                rejected.append({**pattern, 'risk_rejected_reason': reason,
                                 'risk_detail': f"concurrent={concurrent_count} >= max={risk_cfg.MAX_CONCURRENT_POSITIONS}"})
                continue

            # ── Check 3: entry price is zero / unknown ───────────────────────────
            if entry_price <= 0:
                reason = 'entry_price_unknown'
                rejection_reasons[reason] += 1
                rejected.append({**pattern, 'risk_rejected_reason': reason,
                                 'risk_detail': f"entry_price={entry_price}"})
                continue

            # ── Check 4: position sizing floor: can risk_per_trade afford ≥1 share? ──
            # stop_distance = entry * stop_loss_pct_of_account / 100 is a proxy;
            # real sizing = risk_per_trade / stop_distance_per_share
            # We use ATR stop if enabled, else fall back to a 2% stop proxy.
            if risk_cfg.ATR_TRAILING_ENABLED:
                # ATR stop distance is (ATR * multiplier). Without live ATR we can't
                # compute exactly, but we can flag when entry_price alone makes
                # even a 1% stop un-fundable.
                proxy_stop_pct = max(0.01, risk_cfg.ATR_TRAILING_MULTIPLIER * 0.01)
            else:
                proxy_stop_pct = 0.02   # 2% hard-stop fallback
            stop_distance_per_share = entry_price * proxy_stop_pct
            if stop_distance_per_share <= 0:
                reason = 'position_sizing_zero_stop'
                rejection_reasons[reason] += 1
                rejected.append({**pattern, 'risk_rejected_reason': reason,
                                 'risk_detail': f"stop_distance_per_share={stop_distance_per_share:.4f}"})
                continue

            shares = int(risk_per_trade / stop_distance_per_share)
            position_value = shares * entry_price
            if shares < 1:
                reason = 'position_sizing_too_small'
                rejection_reasons[reason] += 1
                rejected.append({**pattern, 'risk_rejected_reason': reason,
                                 'risk_detail': f"capital={initial_capital:.0f} risk_per_trade={risk_per_trade:.2f} "
                                                f"stop_dist={stop_distance_per_share:.4f} → shares={shares}"})
                continue

            if position_value > max_pos_value:
                # Clip shares to max position size rather than reject
                shares = int(max_pos_value / entry_price)
                position_value = shares * entry_price
                if shares < 1:
                    reason = 'position_sizing_exceeds_max_capped_to_zero'
                    rejection_reasons[reason] += 1
                    rejected.append({**pattern, 'risk_rejected_reason': reason,
                                     'risk_detail': f"max_pos_value={max_pos_value:.2f} entry={entry_price:.2f} → 0 shares"})
                    continue

            # ── Check 5: ATR trailing stop feasibility ───────────────────────────
            # Flag if ATR trailing is enabled but pattern has no ATR data attached
            if risk_cfg.ATR_TRAILING_ENABLED:
                bars = pattern.get('bars')
                if bars is not None and not bars.empty:
                    period = risk_cfg.ATR_TRAILING_PERIOD
                    if len(bars) < period:
                        reason = 'atr_insufficient_bars'
                        rejection_reasons[reason] += 1
                        rejected.append({**pattern, 'risk_rejected_reason': reason,
                                         'risk_detail': f"bars={len(bars)} < atr_period={period}"})
                        continue

            # ── Check 6: daily loss limit (simulated) ────────────────────────────
            worst_case_loss = risk_per_trade   # 1 full stop-out
            if simulated_daily_loss + worst_case_loss > daily_loss_limit:
                reason = 'daily_loss_limit_projected'
                rejection_reasons[reason] += 1
                rejected.append({**pattern, 'risk_rejected_reason': reason,
                                 'risk_detail': f"projected_loss={simulated_daily_loss + worst_case_loss:.2f} "
                                                f"> limit={daily_loss_limit:.2f}"})
                continue

            # ── Passed all risk checks ───────────────────────────────────────────
            concurrent_count += 1
            simulated_daily_loss += worst_case_loss   # pessimistic: treat each trade as a stop-out
            tradeable.append({**pattern,
                               'risk_shares': shares,
                               'risk_position_value': round(position_value, 2),
                               'risk_stop_distance': round(stop_distance_per_share, 4)})

        self.logger.info(f"  • Risk: {len(tradeable)} tradeable | {len(rejected)} rejected — {dict(rejection_reasons)}")
        return {'tradeable': tradeable, 'rejected': rejected, 'rejection_reasons': rejection_reasons}

    def _capture_config_snapshot(self):
        cfg = self.config
        self.results['config_snapshot'] = {
            # Screening
            'screening_min_gap_percent':               cfg.screening.MIN_GAP_PERCENT,
            'screening_min_price':                     cfg.screening.MIN_PRICE,
            'screening_min_absolute_volume':           cfg.screening.MIN_ABSOLUTE_VOLUME,
            'screening_min_daily_volume':              cfg.screening.MIN_DAILY_VOLUME,
            'screening_min_cumulative_volume':         cfg.screening.MIN_CUMULATIVE_VOLUME,
            'screening_enable_daily_volume_prescreen': cfg.screening.ENABLE_DAILY_VOLUME_PRESCREEN,
            'screening_min_relative_volume':           cfg.screening.MIN_RELATIVE_VOLUME,
            'screening_enable_relative_volume':        cfg.screening.ENABLE_RELATIVE_VOLUME,
            'screening_cumulative_volume_enabled':     cfg.screening.CUMULATIVE_VOLUME,
            # Session
            'session_premarket_enabled':               cfg.session.PREMARKET_ENABLED,
            'session_premarket_start_et':              cfg.session.PREMARKET_START_ET,
            'session_premarket_end_et':                cfg.session.PREMARKET_END_ET,
            'session_premarket_warmup_minutes':        cfg.session.PREMARKET_WARMUP_MINUTES,
            'session_premarket_min_bars':              cfg.session.PREMARKET_MIN_BARS,
            'session_regular_start_et':                cfg.session.REGULAR_START_ET,
            'session_regular_end_et':                  cfg.session.REGULAR_END_ET,
            'session_regular_warmup_minutes':          cfg.session.REGULAR_WARMUP_MINUTES,
            'session_regular_min_bars':                cfg.session.REGULAR_MIN_BARS,
            'session_after_hours_end_et':              cfg.session.AFTER_HOURS_END_ET,
            # Pattern
            'pattern_min_step_ups':                    cfg.pattern.MIN_STEP_UPS,
            'pattern_min_advance_retention':           cfg.pattern.MIN_ADVANCE_RETENTION,
            'pattern_max_pullback_percent':            cfg.pattern.MAX_PULLBACK_PERCENT,
            'pattern_confluence_normal_min_score':     cfg.pattern.CONFLUENCE_NORMAL_GAP_MIN_SCORE,
            'pattern_confluence_large_min_score':      cfg.pattern.CONFLUENCE_LARGE_GAP_MIN_SCORE,
            'pattern_confluence_extreme_min_score':    cfg.pattern.CONFLUENCE_EXTREME_GAP_MIN_SCORE,
            'pattern_confluence_min_patterns':         cfg.pattern.CONFLUENCE_MIN_PATTERNS,
            'pattern_weight_step_up':                  cfg.pattern.CONFLUENCE_WEIGHT_STEP_UP,
            'pattern_weight_volume':                   cfg.pattern.CONFLUENCE_WEIGHT_VOLUME,
            'pattern_weight_support_resistance':       cfg.pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE,
            # Risk
            'risk_stop_loss_percent_of_account':       cfg.risk.STOP_LOSS_PERCENT_OF_ACCOUNT,
            'risk_max_hold_time_minutes':              cfg.risk.MAX_HOLD_TIME_MINUTES,
            'risk_max_position_size_percent':          cfg.risk.MAX_POSITION_SIZE_PERCENT,
            'risk_max_daily_loss_percent':             cfg.risk.MAX_DAILY_LOSS_PERCENT,
            'risk_max_concurrent_positions':           cfg.risk.MAX_CONCURRENT_POSITIONS,
            'risk_atr_trailing_enabled':               cfg.risk.ATR_TRAILING_ENABLED,
            'risk_atr_trailing_multiplier':            cfg.risk.ATR_TRAILING_MULTIPLIER,
            'risk_atr_trailing_period':                cfg.risk.ATR_TRAILING_PERIOD,
            'risk_breakeven_threshold_pct':            cfg.risk.BREAKEVEN_THRESHOLD_PCT,
            # Reentry
            'reentry_enabled':                         cfg.reentry.ENABLE_REENTRY,
            'reentry_max_per_stock':                   cfg.reentry.MAX_REENTRIES_PER_STOCK,
            'reentry_cooldown_minutes':                cfg.reentry.REENTRY_COOLDOWN_MINUTES,
            # Backtest
            'backtest_start_date':                     cfg.backtest.START_DATE,
            'backtest_end_date':                       cfg.backtest.END_DATE,
            'backtest_initial_capital':                cfg.backtest.INITIAL_CAPITAL,
            'backtest_base_data_dir':                  cfg.backtest.BASE_DATA_DIR,
            'backtest_data_dir':                       cfg.backtest.DATA_DIR,
            'backtest_max_candidates_per_day':         cfg.backtest.MAX_CANDIDATES_PER_DAY,
            'backtest_analysis_window_enabled':        cfg.backtest.ANALYSIS_WINDOW_ENABLED,
            'backtest_analysis_window_start_et':       cfg.backtest.ANALYSIS_WINDOW_START_ET,
            'backtest_analysis_window_end_et':         cfg.backtest.ANALYSIS_WINDOW_END_ET,
            'backtest_simple_stops':                   cfg.backtest.SIMPLE_STOPS,
            'backtest_fast_mode':                      cfg.backtest.FAST_MODE,
        }

    def _log_daily_summary(self, daily_summary: Dict):
        date = daily_summary['date']
        s = daily_summary['stage']

        universe_total = s.get('universe', {}).get('total_universe', 0)
        gap_total      = s.get('gaps', {}).get('total_gaps', 0)
        screened       = len(s.get('screening', {}).get('passed_stocks', []))
        mkt_blocked    = s.get('screening', {}).get('filter_results', {}).get('market_context_blocked', False)
        warmup_pass    = len(s.get('warmup', {}).get('passed_warmup', []))
        pattern_valid  = len(s.get('patterns', {}).get('valid_patterns', []))
        tradeable      = len(s.get('risk', {}).get('tradeable', []))

        self.logger.info("")
        self.logger.info(f"  ── DAILY SUMMARY {date} ──")
        self.logger.info(
            f"  Universe: {universe_total:,} → Gaps: {gap_total:,} → Screened: {screened}"
            f"{' [MKT BLOCKED]' if mkt_blocked else ''}"
            f" → Warmup: {warmup_pass} → Patterns: {pattern_valid} → Tradeable: {tradeable}"
        )
        self.logger.info("  ────────────────────────────────────────")

    def _generate_comprehensive_report(self) -> pd.DataFrame:
        rows = []
        for ds in self.results['daily_summaries']:
            s = ds['stage']
            fr = s.get('screening', {}).get('filter_results', {})
            warmup_fails = Counter(f['reason'] for f in s.get('warmup', {}).get('failed_warmup', []))
            pattern_fails = dict(s.get('patterns', {}).get('failure_reasons', Counter()))

            rows.append({
                'date': ds['date'],
                'universe_total': s.get('universe', {}).get('total_universe', 0),
                'gaps_total': s.get('gaps', {}).get('total_gaps', 0),
                'after_gap_filter': fr.get('after_gap_threshold', 0),
                'after_price_filter': fr.get('after_price_range', 0),
                'after_volume_filter': fr.get('after_volume', 0),
                'after_fundamentals_filter': fr.get('after_float_marketcap', 0),
                'after_rvol_filter': fr.get('after_rvol', 0),
                'market_context_blocked': fr.get('market_context_blocked', False),
                'warmup_passed': len(s.get('warmup', {}).get('passed_warmup', [])),
                'warmup_failed': len(s.get('warmup', {}).get('failed_warmup', [])),
                'patterns_valid': len(s.get('patterns', {}).get('valid_patterns', [])),
                'patterns_failed': len(s.get('patterns', {}).get('failed_patterns', [])),
                'tradeable': len(s.get('risk', {}).get('tradeable', [])),
                'warmup_fail_reasons': str(dict(warmup_fails.most_common(3))),
                'pattern_fail_reasons': str(sorted(pattern_fails.items(), key=lambda x: -x[1])[:3]),
                'gap_min': s.get('gaps', {}).get('gap_distribution', {}).get('min', None),
                'gap_max': s.get('gaps', {}).get('gap_distribution', {}).get('max', None),
                'gap_median': s.get('gaps', {}).get('gap_distribution', {}).get('median', None),
            })

        df = pd.DataFrame(rows)

        if not df.empty:
            self.logger.info("")
            self.logger.info("=" * 100)
            self.logger.info("AGGREGATE REPORT SUMMARY")
            self.logger.info(f"  Days processed              : {len(df)}")
            self.logger.info(f"  Market context days blocked : {df['market_context_blocked'].sum()}")
            self.logger.info(f"  Total tradeable signals     : {df['tradeable'].sum()}")
            self.logger.info(f"  Avg patterns valid / day    : {df['patterns_valid'].mean():.1f}")
            self.logger.info(f"  Avg warmup failed / day     : {df['warmup_failed'].mean():.1f}")
            self.logger.info("=" * 100)

        return df


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive Trade Diagnostic')
    parser.add_argument('--start', default='2024-01-03', help='Start date YYYY-MM-DD')
    parser.add_argument('--end',   default='2024-01-31', help='End date YYYY-MM-DD')
    args = parser.parse_args()

    config = build_config()
    diagnostic = ComprehensiveTradeDiagnostic(config)
    report_df = diagnostic.analyze_date_range(args.start, args.end)

    # ── Timestamped report folder ─────────────────────────────────────────────
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    reports_root = Path(config.backtest.BASE_DATA_DIR) / "reports" / f"diagnostic_{run_ts}"
    reports_root.mkdir(parents=True, exist_ok=True)

    # 1. Daily funnel summary
    report_df.to_csv(reports_root / "daily_summary.csv", index=False)

    # 2. Warmup failures
    warmup_rows = []
    for ds in diagnostic.results['daily_summaries']:
        for f in ds['stage'].get('warmup', {}).get('failed_warmup', []):
            warmup_rows.append({'date': ds['date'], **f})
    pd.DataFrame(warmup_rows).to_csv(reports_root / "warmup_failures.csv", index=False)

    # 3. Pattern failures
    pattern_rows = []
    for ds in diagnostic.results['daily_summaries']:
        for f in ds['stage'].get('patterns', {}).get('failed_patterns', []):
            row = {k: v for k, v in f.items() if k not in ('bars', 'bars_warm')}
            pattern_rows.append({'date': ds['date'], **row})
    pd.DataFrame(pattern_rows).to_csv(reports_root / "pattern_failures.csv", index=False)

    # 4. Valid patterns
    valid_rows = []
    for ds in diagnostic.results['daily_summaries']:
        for p in ds['stage'].get('patterns', {}).get('valid_patterns', []):
            row = {k: v for k, v in p.items() if k not in ('bars', 'bars_warm')}
            valid_rows.append({'date': ds['date'], **row})
    pd.DataFrame(valid_rows).to_csv(reports_root / "valid_patterns.csv", index=False)

    # 5. Config snapshot
    pd.DataFrame([diagnostic.results['config_snapshot']]).to_csv(reports_root / "config_snapshot.csv", index=False)

    # 6. Risk rejections
    risk_rows = []
    for ds in diagnostic.results['daily_summaries']:
        for r in ds['stage'].get('risk', {}).get('rejected', []):
            row = {k: v for k, v in r.items() if k not in ('bars', 'bars_warm')}
            risk_rows.append({'date': ds['date'], **row})
    pd.DataFrame(risk_rows).to_csv(reports_root / "risk_rejections.csv", index=False)

    print(f"\n✓ Reports saved to: {reports_root}")
    print(f"   • daily_summary.csv      — {len(report_df)} days")
    print(f"   • warmup_failures.csv    — {len(warmup_rows)} rejections")
    print(f"   • pattern_failures.csv   — {len(pattern_rows)} rejections")
    print(f"   • valid_patterns.csv     — {len(valid_rows)} passes")
    print(f"   • config_snapshot.csv    — effective config at run time")
    print(f"   • risk_rejections.csv    — {len(risk_rows)} rejections")
    print()
    print(report_df.to_string(index=False))