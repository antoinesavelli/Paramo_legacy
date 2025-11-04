# =====================================================
# screener.backtest.py - Screener adapter for Backtester
# =====================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import collections

from utils.logging import get_logger
from core.risk_manager import calc_atr_stop
from core.pattern_analyzer import PatternAnalyzer
from news.backtest import NewsIntegrationBacktest
from screener.rules import cfg_view_from, filter_price_and_gap

@dataclass
class CandidateSignal:
    symbol: str
    entry_ts: pd.Timestamp
    entry_price: float
    stop_price: float
    gap_percent: float
    pattern_strength: float
    meta: Dict

class BacktestScreener:
    """Date-scoped screener for backtests using LocalDataHandler and pure rules."""
    def __init__(self, config, data_handler, pattern_analyzer: PatternAnalyzer, news_integration: Optional[NewsIntegrationBacktest] = None):
        self.config = config
        self.data = data_handler
        self.pa = pattern_analyzer
        self.news = news_integration
        self.logger = get_logger(__name__, component="bt_screener")

    def calculate_gaps_vectorized(self, day_df: pd.DataFrame, 
                      prev_df: pd.DataFrame,
                      premarket_enabled: bool = False) -> pd.DataFrame:
        """Calculate gaps for all symbols at once with premarket support."""

        if day_df.empty or prev_df.empty:
            return pd.DataFrame()

        # Define time windows based on premarket flag
        day = pd.Timestamp(day_df['timestamp'].iloc[0]).normalize()

        if premarket_enabled:
            session_cfg = self.config.backtest.SESSION
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            open_time_et = pd.Timestamp(day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern')
        else:
            open_time_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')

        open_time_utc = open_time_et.tz_convert('UTC')

        prev_day = day - pd.Timedelta(days=1)
        close_time_et = pd.Timestamp(prev_day.replace(hour=16, minute=0), tz='US/Eastern')
        close_time_utc = close_time_et.tz_convert('UTC')

        # Filter to relevant time windows
        today_open_bars = day_df[day_df['timestamp'] >= open_time_utc]
        prev_close_bars = prev_df[prev_df['timestamp'] <= close_time_utc]

        # Get first bar at or after market open
        today_open = today_open_bars.groupby('symbol')['open'].first().reset_index()
        today_open.columns = ['symbol', 'today_open']

        # Get last bar at or before market close
        prev_close = prev_close_bars.groupby('symbol')['close'].last().reset_index()
        prev_close.columns = ['symbol', 'prev_close']

        # Merge and calculate gaps
        gaps = pd.merge(today_open, prev_close, on='symbol', how='inner')
        gaps['gap_percent'] = ((gaps['today_open'] - gaps['prev_close']) / 
                               gaps['prev_close'] * 100)

        # Use ALL bars for current stats, not just volume > 0
        # Zero-volume bars are legitimate (no trades in that minute)
        current_stats = day_df.groupby('symbol').agg({
            'close': 'last',
            'volume': 'sum'  # Sum will still work correctly with zeros
        }).reset_index()
        current_stats.columns = ['symbol', 'current_price', 'volume']

        gaps = pd.merge(gaps, current_stats, on='symbol', how='left')

        return gaps

    def screen_batch(self, day: datetime) -> List[Dict]:
        """Screen all symbols in parallel."""
        
        # Get day data
        day_str = day.strftime('%Y-%m-%d')
        day_df = self.data_handler.get_day_df_filtered(
            day_str,
            min_price=self.config.screening.MIN_PRICE,
            max_price=self.config.screening.MAX_PRICE,
            min_volume=self.config.screening.MIN_ABSOLUTE_VOLUME
        )
        
        if day_df.empty:
            return []
        
        # Get previous day data
        prev_day = day - timedelta(days=1)
        prev_str = prev_day.strftime('%Y-%m-%d')
        prev_df = self.data_handler._get_day_df(prev_str)
        
        if prev_df.empty:
            return []
        
        # Calculate all gaps at once
        gaps = self.calculate_gaps_vectorized(day_df, prev_df)
        
        # Apply screening filters vectorized
        candidates = gaps[
            (gaps['gap_percent'] >= self.config.screening.MIN_GAP_PERCENT) &
            (gaps['volume'] >= self.config.screening.MIN_ABSOLUTE_VOLUME) &
            (gaps['current_price'] >= self.config.screening.MIN_PRICE) &
            (gaps['current_price'] <= self.config.screening.MAX_PRICE)
        ]
        
        # Sort by gap percentage
        candidates = candidates.sort_values('gap_percent', ascending=False)
        
        return candidates.to_dict('records')

    def screen(self, day: datetime) -> Dict[str, List]:
        """Screen for candidates with premarket support based on config."""
        
        diagnostics: List[Dict] = []
        signals: List[CandidateSignal] = []

        stats = {
            "universe_size": 0,
            "gap_rows": 0,
            "after_price_gap_filter": 0,
            "processed": 0,
            "pattern_valid": 0,
            "signals": 0,
            "reject_reasons": collections.Counter()
        }
        
        # ✅ FIXED: Define progress_every from config
        progress_every = self.config.system.LOG_PROGRESS_EVERY_N_SYMBOLS

        # ✅ FIXED: Define log_progress function
        def log_progress(final: bool = False):
            if stats["processed"] == 0:
                return
            top_rej = stats["reject_reasons"].most_common(3)
            tag = "FINAL" if final else "PROGRESS"
            self.logger.info(
                f"[{tag}] {day.date()} | "
                f"Processed: {stats['processed']}/{stats['after_price_gap_filter']} | "
                f"Signals: {stats['signals']} | "
                f"Pattern Valid: {stats['pattern_valid']} | "
                f"Top Rejects: {top_rej}"
            )

        # OPTIMIZATION 1: Check current day exists BEFORE any logging
        day_str = day.strftime('%Y-%m-%d')
        if not self.data.has_data_for_date(day_str):
            # Skip silently or with minimal logging
            return {"signals": signals, "diagnostics": diagnostics, "stats": stats}

        # OPTIMIZATION 2: Check if previous trading day exists (within 7 days)
        prev_day = day - timedelta(days=1)
        found_prev = False
        for lookback in range(7):
            prev_str = prev_day.strftime('%Y-%m-%d')
            if self.data.has_data_for_date(prev_str):
                found_prev = True
                break
            prev_day = prev_day - timedelta(days=1)
        
        if not found_prev:
            # No previous trading day within 7 days - skip immediately
            return {"signals": signals, "diagnostics": diagnostics, "stats": stats}

        # NOW start logging (only for days we'll actually process)
        self.logger.info("=" * 80)
        self.logger.info(f"SCREENING DAY: {day.date()}")
        self.logger.info("=" * 80)

        uni = self.data.get_universe()
        stats["universe_size"] = len(uni)
        if uni.empty:
            self.logger.warning(f"[SKIP] {day.date()} - Universe empty")
            return {"signals": signals, "diagnostics": diagnostics, "stats": stats}
        
        self.logger.info(f"[DATA] Universe size: {stats['universe_size']} symbols")
        
        session_cfg = self.config.backtest.SESSION
        premarket_enabled = session_cfg.PREMARKET_ENABLED
        
        # Log session configuration
        self.logger.info(f"[CONFIG] Premarket: {premarket_enabled} | "
                        f"Min Gap: {session_cfg.PREMARKET_MIN_GAP_PERCENT if premarket_enabled else self.config.screening.MIN_GAP_PERCENT}% | "
                        f"Warmup: {session_cfg.PREMARKET_WARMUP_MINUTES if premarket_enabled else self.config.backtest.WARMUP_MINUTES} mins")
        
        # Calculate gaps with automatic previous day lookup
        self.logger.info(f"[GAPS] Calculating gaps for {day.date()}...")
        gaps_info = self.data.calculate_gaps(day, premarket=premarket_enabled)
        gaps_df: pd.DataFrame = gaps_info.get('gaps', pd.DataFrame())
        stats["gap_rows"] = 0 if gaps_df is None else len(gaps_df)
        prev_syms = gaps_info.get('prev_syms', set())
        today_syms = gaps_info.get('today_syms', set())
        prev_empty = gaps_info.get('prev_empty', False)
        today_empty = gaps_info.get('today_empty', False)
        lookback_days = gaps_info.get('lookback_days', 0)
        prev_day_used = gaps_info.get('prev_day_used', 'unknown')

        # Log gap calculation results
        if lookback_days > 0:
            self.logger.info(
                f"[GAPS] Using previous trading day: {prev_day_used} ({lookback_days} days back)"
            )
        
        self.logger.info(
            f"[GAPS] Found {stats['gap_rows']} symbols with gaps | "
            f"Previous day: {len(prev_syms)} symbols | Today: {len(today_syms)} symbols"
        )

        if prev_empty or today_empty:
            if lookback_days >= 7:
                self.logger.warning(
                    f"[SKIP] {day.date()} - No previous trading day found within 7 days"
                )
            else:
                self.logger.info(
                    f"[SKIP] {day.date()} - Missing day data (prev_empty={prev_empty}, today_empty={today_empty})"
                )
            return {"signals": signals, "diagnostics": diagnostics, "stats": stats}

        only_prev = sorted(prev_syms - today_syms)
        only_today = sorted(today_syms - prev_syms)
        
        if len(only_prev) > 0:
            stats["reject_reasons"]["no_open_bar"] += len(only_prev)
            self.logger.debug(f"[GAPS] {len(only_prev)} symbols missing open bar")
        if len(only_today) > 0:
            stats["reject_reasons"]["no_prev_close"] += len(only_today)
            self.logger.debug(f"[GAPS] {len(only_today)} symbols missing previous close")

        if gaps_df.empty:
            self.logger.warning(f"[SKIP] {day.date()} - No matching symbols after gap calculation")
            return {"signals": signals, "diagnostics": diagnostics, "stats": stats}

        # Apply filters
        cfg = cfg_view_from(self.config)
        
        if premarket_enabled:
            min_gap = session_cfg.PREMARKET_MIN_GAP_PERCENT
            min_volume = session_cfg.PREMARKET_MIN_ABSOLUTE_VOLUME
        else:
            min_gap = cfg.MIN_GAP_PERCENT
            min_volume = cfg.MIN_ABSOLUTE_VOLUME
        
        pre_filter = len(gaps_df)
        base = gaps_df[
            (gaps_df['gap_percent'].abs() >= min_gap) &
            (gaps_df['last_price'] >= cfg.MIN_PRICE) &
            (gaps_df['last_price'] <= cfg.MAX_PRICE)
        ]
        stats["after_price_gap_filter"] = len(base)
        
        # Log filtering results
        filtered_out = pre_filter - len(base)
        self.logger.info(
            f"[FILTER] Applied gap/price filters | "
            f"Before: {pre_filter} | After: {len(base)} | Filtered: {filtered_out}"
        )
        
        if base.empty:
            self.logger.warning(f"[SKIP] {day.date()} - All symbols filtered out")
            stats["reject_reasons"]["gap_price_filter_all"] += 1
            return {"signals": signals, "diagnostics": diagnostics, "stats": stats}

        # Define session boundaries
        if premarket_enabled:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            pm_end = datetime.strptime(session_cfg.PREMARKET_END_ET, "%H:%M").time()
            
            session_start_et = pd.Timestamp(day.replace(hour=pm_start.hour, minute=pm_start.minute), tz='US/Eastern')
            premarket_end_et = pd.Timestamp(day.replace(hour=pm_end.hour, minute=pm_end.minute), tz='US/Eastern')
            warmup = session_cfg.PREMARKET_WARMUP_MINUTES
        else:
            session_start_et = pd.Timestamp(day.replace(hour=9, minute=30), tz='US/Eastern')
            premarket_end_et = session_start_et
            warmup = getattr(self.config.backtest, "WARMUP_MINUTES", 45)
        
        session_end_et = pd.Timestamp(day.replace(hour=20, minute=0), tz='US/Eastern')
        session_start_utc = session_start_et.tz_convert('UTC')
        premarket_end_utc = premarket_end_et.tz_convert('UTC')
        session_end_utc = session_end_et.tz_convert('UTC')
        
        min_news_strength = getattr(self.config.backtest, "MIN_NEWS_STRENGTH", 30)
        ignore_catalyst = getattr(self.config.backtest, "IGNORE_CATALYST", True)

        base = base.sort_values('gap_percent', ascending=False)
        
        # Log top gappers
        top_5 = base.head(5)
        self.logger.info(f"[GAPPERS] Top 5 by gap %:")
        for idx, row in top_5.iterrows():
            self.logger.info(
                f"  • {row['symbol']}: {row['gap_percent']:+.2f}% | "
                f"Price: ${row['last_price']:.2f} | Vol: {row.get('volume', 0):,}"
            )

        # Start pattern analysis
        self.logger.info(f"[ANALYSIS] Analyzing {len(base)} candidates for patterns...")

        for idx, (_, r) in enumerate(base.iterrows(), 1):
            symbol = str(r['symbol'])
            stats["processed"] += 1

            # Optional news gate
            if not ignore_catalyst and self.news:
                try:
                    news = self.news.analyze_news_impact(symbol, date=day)
                except Exception as e:
                    self.logger.debug(f"[NEWS] {symbol} - Analysis error: {e}")
                    news = {}
                if not news.get('has_catalyst') or int(news.get('catalyst_strength', 0)) < min_news_strength:
                    diagnostics.append({"date": pd.Timestamp(day).date(), "symbol": symbol, "phase": "reject", "reason": "weak_catalyst", "catalyst_strength": news.get('catalyst_strength')})
                    stats["reject_reasons"]["weak_catalyst"] += 1
                    continue

            bars = self.data.get_intraday_bars(symbol, start=session_start_utc, end=session_end_utc)
            if bars.empty:
                diagnostics.append({"date": pd.Timestamp(day).date(), "symbol": symbol, "phase": "reject", "reason": "empty_bars"})
                stats["reject_reasons"]["empty_bars"] += 1
                continue
            
            # Apply bar count check
            if premarket_enabled:
                min_bars = session_cfg.PREMARKET_MIN_BARS
                if len(bars) < min_bars:
                    diagnostics.append({"date": pd.Timestamp(day).date(), "symbol": symbol, "phase": "reject", "reason": "premarket_min_bars", "rows": len(bars), "min_bars": min_bars})
                    stats["reject_reasons"]["premarket_min_bars"] += 1
                    continue
            
            if len(bars) < warmup:
                diagnostics.append({"date": pd.Timestamp(day).date(), "symbol": symbol, "phase": "reject", "reason": "warmup_short", "rows": len(bars), "warmup": warmup})
                stats["reject_reasons"]["warmup_short"] += 1
                continue

            missing_cols = [cname for cname in ['open', 'high', 'low', 'close', 'volume', 'timestamp'] if cname not in bars.columns]
            if missing_cols:
                diagnostics.append({"date": pd.Timestamp(day).date(), "symbol": symbol, "phase": "reject", "reason": "bars_missing_columns", "missing": ",".join(missing_cols)})
                stats["reject_reasons"]["bars_missing_columns"] += 1
                continue
            if bars.iloc[:warmup].isna().any().any():
                diagnostics.append({"date": pd.Timestamp(day).date(), "symbol": symbol, "phase": "reject", "reason": "nan_in_warmup"})
                stats["reject_reasons"]["nan_in_warmup"] += 1
                continue

            bars_warm = bars.iloc[:warmup].copy()
            pa = self.pa.analyze_pattern(symbol, bars=bars_warm, is_premarket=premarket_enabled)
            
            if not pa.get('valid', False):
                step_up = pa.get('step_ups', {}) or {}
                parabolic = pa.get('parabolic', {}) or {}
                breakout = pa.get('breakout', {}) or {}
                volume = pa.get('volume', {}) or {}
                sr = pa.get('support_resistance', {}) or {}
                
                # Detailed pattern rejection logging
                if self.config.backtest.LOG_PATTERN_ANALYSIS:
                    self.logger.debug(
                        f"[PATTERN] {symbol} REJECTED | "
                        f"Score: {pa.get('pattern_strength', 0):.1f} | "
                        f"Reason: {pa.get('reason')} | "
                        f"Detected: {pa.get('patterns_detected', [])}"
                    )
                
                diagnostics.append({
                    "date": pd.Timestamp(day).date(), "symbol": symbol, "phase": "reject", "reason": "pattern_invalid",
                    "pattern_strength": pa.get('pattern_strength'), "pattern_count": pa.get('pattern_count'),
                    "patterns_detected": "|".join(pa.get('patterns_detected', [])),
                    "pa_reason": pa.get('reason'),
                    "step_up_detected": step_up.get('detected'), "step_up_steps": step_up.get('step_count'), "step_up_retention": step_up.get('retention_rate'),
                    "parabolic_detected": parabolic.get('detected'), "parabolic_angle": parabolic.get('angle'), "parabolic_accel": parabolic.get('acceleration'), "parabolic_vol_mult": parabolic.get('volume_multiplier'),
                    "breakout_detected": breakout.get('detected'), "breakout_level": breakout.get('breakout_level'), "breakout_vol_ratio": breakout.get('volume_ratio'),
                    "volume_strength": volume.get('strength'), "sr_strength": sr.get('strength')
                })
                stats["reject_reasons"]["pattern_invalid"] += 1
                continue

            stats["pattern_valid"] += 1
            entry_row = bars.iloc[warmup - 1]
            entry_price = float(entry_row['close'])
            entry_ts = entry_row['timestamp']
            
            is_premarket_entry = premarket_enabled and entry_ts < premarket_end_utc
            
            stop_price = calc_atr_stop(bars_warm, entry_price, atr_period=14, atr_mult=2.0, fallback_pct=0.03)
            risk_ps = entry_price - stop_price
            
            # Log valid signals
            self.logger.info(
                f"[SIGNAL] {symbol} | "
                f"Gap: {float(r['gap_percent']):+.2f}% | "
                f"Entry: ${entry_price:.2f} | "
                f"Stop: ${stop_price:.2f} | "
                f"Risk: ${risk_ps:.2f} | "
                f"Pattern Score: {float(pa.get('pattern_strength') or 0):.1f} | "
                f"Premarket: {is_premarket_entry}"
            )

            signals.append(CandidateSignal(
                symbol=symbol,
                entry_ts=entry_ts,
                entry_price=entry_price,
                stop_price=stop_price,
                gap_percent=float(r['gap_percent']),
                pattern_strength=float(pa.get('pattern_strength', 0) or 0),
                meta={
                    "last_price": float(r['last_price']), 
                    "open_price": float(r['open_price']), 
                    "prev_close": float(r['prev_close']),
                    "is_premarket": is_premarket_entry
                }
            ))
            stats["signals"] += 1

            # Log progress periodically
            if stats["processed"] % progress_every == 0:
                log_progress()

        # Final daily summary
        log_progress(final=True)

        # End of day summary
        self.logger.info("=" * 80)
        self.logger.info(f"DAY COMPLETE: {day.date()}")
        self.logger.info(f"  • Candidates Analyzed: {stats['processed']}")
        self.logger.info(f"  • Signals Generated: {stats['signals']}")
        self.logger.info(f"  • Pattern Valid: {stats['pattern_valid']}")
        self.logger.info(f"  • Top Rejection Reasons:")
        for reason, count in stats["reject_reasons"].most_common(3):
            self.logger.info(f"    - {reason}: {count}")
        self.logger.info("=" * 80)

        diagnostics.append({
            "date": pd.Timestamp(day).date(),
            "symbol": "__DAY__",
            "phase": "summary",
            "reason": "screen_summary",
            "premarket_enabled": premarket_enabled,
            "universe_size": stats["universe_size"],
            "gap_rows": stats["gap_rows"],
            "after_price_gap_filter": stats["after_price_gap_filter"],
            "processed": stats["processed"],
            "pattern_valid": stats["pattern_valid"],
            "signals": stats["signals"],
            "top_rejects": dict(stats["reject_reasons"].most_common(5))
        })

        return {"signals": signals, "diagnostics": diagnostics, "stats": stats}
