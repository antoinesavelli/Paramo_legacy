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

# Add repository root to path (scripts/analysis -> repo root)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.loader import build_config
from data_handler.local import LocalDataHandler
from data_handler.aggregates.aggregate_handler import AggregateDataHandler
from strategy.patterns.pattern_analyzer import PatternAnalyzer
from screener.helpers import BacktestRelativeVolumeCalculator
from market_context.backtest import BacktestMarketContext
from news.backtest import NewsIntegrationBacktest
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

        agg_dir = Path(config.backtest.DAILY_AGGREGATES_DIR)
        self._agg_handler = AggregateDataHandler(str(agg_dir))

        self._rvol_calculator = BacktestRelativeVolumeCalculator(config, self.data_handler, self.logger)
        self.news_integration = NewsIntegrationBacktest(config, data_dir=config.backtest.NEWS_DATA_DIR)

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
        """Screening filters aligned to UnifiedScreener pre-pattern flow."""
        if gaps_df is None or gaps_df.empty:
            return {'passed_stocks': [], 'filter_results': {}, 'message': 'No gaps to analyze'}

        cfg = self.config.screening
        filter_results = {
            'initial': len(gaps_df),
            'after_gap_threshold': 0,
            'after_daily_volume': 0,
            'after_rvol': 0,
            'after_fundamentals': 0,
            'market_context_blocked': False
        }

        min_gap = cfg.MIN_GAP_PERCENT
        after_gap = gaps_df[gaps_df['gap_percent'] >= min_gap].copy()
        filter_results['after_gap_threshold'] = len(after_gap)
        self.logger.info(f"  • [F1] Gap >= {min_gap}%: {filter_results['initial']} → {filter_results['after_gap_threshold']}")

        after_volume = after_gap.copy()
        if cfg.ENABLE_DAILY_VOLUME_PRESCREEN:
            after_volume = after_volume[after_volume['volume'] >= cfg.MIN_DAILY_VOLUME]
        filter_results['after_daily_volume'] = len(after_volume)
        self.logger.info(
            f"  • [F2] Daily Volume >= {cfg.MIN_DAILY_VOLUME:,} "
            f"(enabled={cfg.ENABLE_DAILY_VOLUME_PRESCREEN}): "
            f"{filter_results['after_gap_threshold']} → {filter_results['after_daily_volume']}"
        )

        after_rvol = after_volume.copy()
        if cfg.ENABLE_RELATIVE_VOLUME:
            min_rvol = cfg.MIN_RELATIVE_VOLUME
            max_rvol_checks = min(
                len(after_rvol),
                self.config.backtest.MAX_CANDIDATES_PER_DAY * 2
            )
            top_symbols = after_rvol.head(max_rvol_checks)['symbol'].tolist()
            try:
                rel_vols = self._rvol_calculator.calculate_batch(top_symbols, day)
                after_rvol['relative_volume'] = after_rvol['symbol'].map(rel_vols).fillna(0.0)
                after_rvol = after_rvol[after_rvol['relative_volume'] >= min_rvol]
            except Exception as e:
                self.logger.warning(f"  • [F3] RVOL calculation error: {e} — filter skipped")
            self.logger.info(f"  • [F3] Relative Volume >= {min_rvol}x: {filter_results['after_daily_volume']} → {len(after_rvol)}")
        else:
            self.logger.info("  • [F3] Relative Volume (disabled)")
        filter_results['after_rvol'] = len(after_rvol)

        after_fundamentals = after_rvol.copy()
        if cfg.ENABLE_FLOAT_FILTER or cfg.ENABLE_MARKETCAP_FILTER:
            day_agg = self._agg_handler.get_day_aggregates(day)
            if day_agg is not None and not day_agg.empty:
                lookup = day_agg.set_index('symbol')

                def passes_fundamentals(symbol: str) -> bool:
                    if symbol not in lookup.index:
                        return False
                    row = lookup.loc[symbol]
                    if cfg.ENABLE_FLOAT_FILTER:
                        if 'float' not in row or pd.isna(row['float']) or row['float'] > cfg.MAX_FLOAT:
                            return False
                    if cfg.ENABLE_MARKETCAP_FILTER:
                        if 'marketcap' not in row or pd.isna(row['marketcap']) or row['marketcap'] > cfg.MAX_MARKETCAP:
                            return False
                    return True

                after_fundamentals = after_rvol[after_rvol['symbol'].apply(passes_fundamentals)]
            else:
                self.logger.warning("  • [F4] Fundamental filters enabled but aggregate data unavailable")
        filter_results['after_fundamentals'] = len(after_fundamentals)
        self.logger.info(
            f"  • [F4] Float/Marketcap (enabled={cfg.ENABLE_FLOAT_FILTER or cfg.ENABLE_MARKETCAP_FILTER}): "
            f"{filter_results['after_rvol']} → {filter_results['after_fundamentals']}"
        )

        if self.market_context is not None:
            try:
                self.market_context.update_market_context(day)
                score = self.market_context.market_indicators.get('market_score', 0)
                env = self.market_context.market_indicators.get('trading_environment', 'unknown')
                if not self.market_context.should_trade():
                    self.logger.warning(f"  • [F5] Market Context BLOCKED — score={score:.1f}, env={env}")
                    filter_results['market_context_blocked'] = True
                    return {'passed_stocks': [], 'filter_results': filter_results}
                self.logger.info(f"  • [F5] Market Context OK — score={score:.1f}, env={env}")
            except Exception as e:
                self.logger.warning(f"  • [F5] Market Context error: {e} — not gating")
        else:
            self.logger.info("  • [F5] Market Context (not available — skipped)")

        passed_stocks = after_fundamentals.to_dict('records')
        if passed_stocks:
            top2 = sorted(passed_stocks, key=lambda x: x['gap_percent'], reverse=True)[:2]
            self.logger.info(
                f"  • {len(passed_stocks)} passed. Top 2: "
                + " | ".join(f"{s['symbol']} {s['gap_percent']:+.1f}%" for s in top2)
            )

        return {'passed_stocks': passed_stocks, 'filter_results': filter_results}

    def _analyze_warmup_requirements(self, day: datetime, stocks: List[Dict]) -> Dict:
        """Warmup checks aligned to UnifiedScreener: bars/NaN/readiness only."""
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

        self.logger.info(f"  • Session Start: {session_start_et.strftime('%H:%M %Z')} | Warmup: {warmup} bars | Min bars: {min_bars}")
        self.logger.info(f"  • Checking {len(stocks)} stocks...")

        passed_warmup = []
        failed_warmup = []
        bar_counts = []
        news_probe_ts = session_start_utc + pd.Timedelta(minutes=warmup)

        for stock in stocks:
            symbol = stock['symbol']

            if not getattr(self.config.backtest, 'IGNORE_CATALYST', True):
                try:
                    news = self.news_integration.check_news_approval(symbol, news_probe_ts)
                except Exception as e:
                    news = {'approved': False, 'reason': f'error:{type(e).__name__}'}

                if not news.get('approved', False):
                    reason = f"news_{news.get('reason', 'unknown')}"
                    failed_warmup.append({'symbol': symbol, 'reason': reason, 'gap_percent': stock['gap_percent']})
                    continue

            bars = self.data_handler.get_intraday_bars(symbol, start=session_start_utc, end=session_end_utc)
            bar_count = len(bars) if bars is not None and not bars.empty else 0
            bar_counts.append(bar_count)

            if bars is None or bars.empty:
                failed_warmup.append({'symbol': symbol, 'reason': 'empty_bars', 'bar_count': 0, 'gap_percent': stock['gap_percent']})
                continue

            if bar_count < warmup:
                failed_warmup.append({'symbol': symbol, 'reason': 'insufficient_bars_for_pattern', 'bar_count': bar_count, 'warmup_required': warmup, 'gap_percent': stock['gap_percent']})
                continue

            if bars.iloc[:warmup].isna().any().any():
                failed_warmup.append({'symbol': symbol, 'reason': 'nan_in_pattern_window',
                                      'bar_count': bar_count, 'gap_percent': stock['gap_percent']})
                continue

            if bar_count <= warmup:
                failed_warmup.append({
                    'symbol': symbol, 'reason': 'no_next_bar_for_entry',
                    'bar_count': bar_count, 'warmup_required': warmup,
                    'gap_percent': stock['gap_percent']
                })
                continue

            stock = stock.copy()
            stock['bar_count'] = bar_count
            stock['bars'] = bars
            stock['bars_warm'] = bars.iloc[:warmup].copy()
            stock['warmup'] = warmup
            stock['signal_bar_close'] = float(bars.iloc[warmup - 1]['close'])
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
        min_price = self.config.screening.MIN_PRICE
        analysis_window_enabled = bool(getattr(self.config.backtest, 'ANALYSIS_WINDOW_ENABLED', False))
        analysis_window_start_utc = None
        analysis_window_end_utc = None
        window_start_str = None
        window_end_str = None
        if analysis_window_enabled:
            window_start_str = getattr(self.config.backtest, 'ANALYSIS_WINDOW_START_ET', '04:00')
            window_end_str = getattr(self.config.backtest, 'ANALYSIS_WINDOW_END_ET', '16:00')
            ws = datetime.strptime(window_start_str, "%H:%M").time()
            we = datetime.strptime(window_end_str, "%H:%M").time()
            analysis_window_start_utc = pd.Timestamp(
                day.replace(hour=ws.hour, minute=ws.minute), tz='US/Eastern'
            ).tz_convert('UTC')
            analysis_window_end_utc = pd.Timestamp(
                day.replace(hour=we.hour, minute=we.minute), tz='US/Eastern'
            ).tz_convert('UTC')

        self.logger.info(f"  • Running pattern analysis on {len(stocks)} stocks...")

        for stock in stocks:
            symbol = stock['symbol']
            gap_percent = stock.get('gap_percent', None)
            try:
                bars = stock.get('bars')
                bars_warm = stock.get('bars_warm')
                warmup = int(stock.get('warmup', 0) or 0)
                if bars_warm is None or bars_warm.empty or bars is None or bars.empty:
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

                if not result.get('valid'):
                    reason = result.get('reason', 'pattern_invalid')
                    failed_patterns.append({
                        **stock,
                        'reason': reason,
                        'pattern_strength': result.get('pattern_strength', 0),
                        'min_score_threshold': result.get('min_score_threshold')
                    })
                    failure_reasons[reason] += 1
                    continue

                fsb_idx = int(result.get('first_signal_bar_idx', -1) or -1)
                if fsb_idx >= 0 and fsb_idx + 1 < len(bars):
                    fill_row = bars.iloc[fsb_idx + 1]
                elif warmup < len(bars):
                    fill_row = bars.iloc[warmup]
                else:
                    reason = 'no_next_bar_for_entry'
                    failed_patterns.append({**stock, 'reason': reason})
                    failure_reasons[reason] += 1
                    continue

                entry_price = float(fill_row['open'])
                if entry_price < min_price:
                    reason = 'price_out_of_range'
                    failed_patterns.append({**stock, 'reason': reason, 'entry_price': entry_price, 'min_price': min_price})
                    failure_reasons[reason] += 1
                    continue

                entry_ts = pd.Timestamp(fill_row['timestamp'])
                entry_ts_utc = entry_ts.tz_localize('UTC') if entry_ts.tz is None else entry_ts.tz_convert('UTC')
                if analysis_window_start_utc is not None and (
                    entry_ts_utc < analysis_window_start_utc or entry_ts_utc >= analysis_window_end_utc
                ):
                    reason = 'outside_analysis_window'
                    failed_patterns.append({
                        **stock,
                        'reason': reason,
                        'entry_time_et': entry_ts_utc.tz_convert('US/Eastern').strftime('%H:%M:%S'),
                        'window_start_et': window_start_str,
                        'window_end_et': window_end_str
                    })
                    failure_reasons[reason] += 1
                    continue

                valid_patterns.append({
                    **stock,
                    'pattern_strength': result.get('pattern_strength'),
                    'patterns_detected': result.get('patterns_detected'),
                    'min_score_threshold': result.get('min_score_threshold'),
                    'entry_price': entry_price,
                    'entry_ts': entry_ts_utc,
                })

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
            'reentry_enabled':                         cfg.risk.ENABLE_REENTRY,
            'reentry_max_per_stock':                   cfg.risk.MAX_REENTRIES_PER_STOCK,
            'reentry_cooldown_minutes':                cfg.risk.REENTRY_COOLDOWN_MINUTES,
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
                'after_daily_volume_filter': fr.get('after_daily_volume', 0),
                'after_rvol_filter': fr.get('after_rvol', 0),
                'after_fundamentals_filter': fr.get('after_fundamentals', 0),
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


# ---────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    from run.backtest_engine import Backtester

    parser = argparse.ArgumentParser(description='Comprehensive Trade Diagnostic')
    parser.add_argument('--start', default=None, help='Start date YYYY-MM-DD (default: config.backtest.START_DATE)')
    parser.add_argument('--end', default=None, help='End date YYYY-MM-DD (default: config.backtest.END_DATE)')
    parser.add_argument('--override', action='append', help='Config override key=value (repeatable)')
    parser.add_argument('--no-env-layer', action='store_true', help='Disable environment variable override layer')
    parser.add_argument('--skip-backtest-compare', action='store_true', help='Skip plain backtest comparison run')
    args = parser.parse_args()

    config = build_config(
        cli_overrides=args.override,
        enable_env_layer=not args.no_env_layer,
    )
    start = args.start or config.backtest.START_DATE
    end = args.end or config.backtest.END_DATE

    diagnostic = ComprehensiveTradeDiagnostic(config)
    report_df = diagnostic.analyze_date_range(start, end)

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    reports_root = Path(config.system.REPORTS_DIR) / f"diagnostic_{run_ts}"
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

    diagnostic_tradeable = int(report_df['tradeable'].sum()) if not report_df.empty else 0
    plain_backtest_trades = None

    if not args.skip_backtest_compare:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        local_data_handler = LocalDataHandler(config, data_dir=config.backtest.DATA_DIR)
        pattern_analyzer = PatternAnalyzer(config, local_data_handler)
        bt = Backtester(config, local_data_handler, pattern_analyzer=pattern_analyzer)
        bt_results = bt.run_backtest(start_dt, end_dt, initial_capital=config.backtest.INITIAL_CAPITAL)
        plain_backtest_trades = int(bt_results.get('statistics', {}).get('total_trades', 0) or 0)

    print(f"\nReports saved to: {reports_root}")
    print(
        f"Summary: days={len(report_df)}, tradeable_signals={diagnostic_tradeable}, "
        f"warmup_rejections={len(warmup_rows)}, pattern_rejections={len(pattern_rows)}"
    )

    if plain_backtest_trades is not None:
        print(
            f"Backtest comparison: diagnostic_tradeable_signals={diagnostic_tradeable}, "
            f"plain_backtest_trades={plain_backtest_trades}"
        )
        if diagnostic_tradeable > 0 and plain_backtest_trades == 0:
            print(
                "Reason hint: diagnostic counts screen/risk-passed candidates, while plain backtest "
                "counts executed-and-closed trades after simulation gates."
            )
        elif diagnostic_tradeable == 0 and plain_backtest_trades == 0:
            print(
                "Reason hint: both flows found zero opportunities in this range; daily_summary.csv "
                "shows where candidates are first filtered out."
            )

    if not report_df.empty:
        print("Example daily rows (max 2):")
        print(report_df.head(2).to_string(index=False))