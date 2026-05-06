# =====================================================
# screener.core.py - Unified screening logic (live + backtest)
# =====================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import collections
from pathlib import Path

from utils.logging import get_logger
from strategy.risk_manager import calc_atr_stop
from strategy.pattern_analyzer import PatternAnalyzer
from screener.rules import cfg_view_from
from screener.helpers import (
    LiveRelativeVolumeCalculator,
    BacktestRelativeVolumeCalculator,
    DiagnosticCreator
)
from data_handler.aggregate_handler import AggregateDataHandler


@dataclass
class CandidateSignal:
    """Unified signal output for both live and backtest."""
    symbol: str
    entry_ts: pd.Timestamp
    entry_price: float
    stop_price: float
    gap_percent: float
    pattern_strength: float
    relative_volume: Optional[float]
    meta: Dict


class UnifiedScreener:
    """Unified screener logic used by both live and backtest."""

    def __init__(
        self,
        config,
        data_handler,
        pattern_analyzer: PatternAnalyzer,
        news_integration=None,
        market_context=None,
        logger=None,
        is_live=False
    ):
        self.config = config
        self.data = data_handler
        self.pa = pattern_analyzer
        self.news = news_integration
        self.market_context = market_context
        self.logger = logger or get_logger(__name__, component="screener")
        self.is_live = is_live

        if is_live:
            self.rvol_calculator = LiveRelativeVolumeCalculator(config, data_handler, self.logger)
        else:
            self.rvol_calculator = BacktestRelativeVolumeCalculator(config, data_handler, self.logger)

        self.diagnostic_creator = DiagnosticCreator(config)

        aggregate_dir = Path(config.backtest.BASE_DATA_DIR) / "daily_aggregates"
        self.aggregate_handler = AggregateDataHandler(str(aggregate_dir))

    def screen_symbols(
        self,
        candidates_df: pd.DataFrame,
        day: datetime,
        is_live: bool = False
    ) -> Dict[str, List]:
        """
        Unified screening logic for both live and backtest.

        Filtering order:
        1. Daily volume pre-screen (from aggregates)
        2. Relative volume filter (if enabled)
        3. Load intraday bars
        4. Gap % confirmed check (during continuous monitoring)
        5. Price range check (current price validation)
        6. Pattern analysis (expensive operation - only for survivors)

        Args:
            candidates_df: DataFrame with gap_percent, last_price, symbol
            day: Trading day (for historical lookups)
            is_live: True for live trading, False for backtest

        Returns:
            {
                "signals": List[CandidateSignal],
                "diagnostics": List[Dict],
                "stats": Dict
            }
        """
        diagnostics: List[Dict] = []
        signals: List[CandidateSignal] = []

        stats = {
            "candidates_in": len(candidates_df),
            "after_daily_volume_filter": 0,
            "after_relative_volume_filter": 0,
            "after_price_filter": 0,
            "processed": 0,
            "pattern_valid": 0,
            "signals": 0,
            "reject_reasons": collections.Counter()
        }

        if candidates_df.empty:
            return {"signals": signals, "diagnostics": diagnostics, "stats": stats}

        # Sort by gap % (highest first)
        candidates_df = candidates_df.sort_values('gap_percent', ascending=False)

        # Fetch aggregate data once for this day — shared by volume and fundamental filters
        # to avoid redundant I/O from each filter method calling get_day_aggregates() independently.
        needs_aggregates = (
            self.config.screening.ENABLE_DAILY_VOLUME_PRESCREEN
            or self.config.screening.ENABLE_FLOAT_FILTER
            or self.config.screening.ENABLE_MARKETCAP_FILTER
        )
        day_aggregates = self.aggregate_handler.get_day_aggregates(day) if needs_aggregates else None

        # Apply daily volume pre-screen using aggregates
        if self.config.screening.ENABLE_DAILY_VOLUME_PRESCREEN:
            candidates_df = self._apply_daily_volume_filter(candidates_df, day, stats, day_aggregates)
            if candidates_df.empty:
                self.logger.warning("All candidates filtered by daily volume pre-screen")
                return {"signals": signals, "diagnostics": diagnostics, "stats": stats}
        else:
            stats["after_daily_volume_filter"] = len(candidates_df)

        # Apply RVOL filter if enabled
        if self.config.screening.ENABLE_RELATIVE_VOLUME:
            candidates_df = self._apply_rvol_filter(candidates_df, day, is_live, stats)
            if candidates_df.empty:
                self.logger.warning("All candidates filtered by relative volume")
                return {"signals": signals, "diagnostics": diagnostics, "stats": stats}
        else:
            stats["after_relative_volume_filter"] = len(candidates_df)

        # Apply fundamental filters (float and marketcap)
        if self.config.screening.ENABLE_FLOAT_FILTER or self.config.screening.ENABLE_MARKETCAP_FILTER:
            candidates_df = self._apply_fundamental_filters(candidates_df, day, stats, day_aggregates)
            if candidates_df.empty:
                self.logger.warning("All candidates filtered by fundamental filters")
                return {"signals": signals, "diagnostics": diagnostics, "stats": stats}
        else:
            stats["after_fundamental_filter"] = len(candidates_df)

        # Get session configuration
        session_cfg = self.config.session
        premarket_enabled = session_cfg.PREMARKET_ENABLED

        # Define session boundaries
        if premarket_enabled:
            pm_start = datetime.strptime(session_cfg.PREMARKET_START_ET, "%H:%M").time()
            pm_end = datetime.strptime(session_cfg.PREMARKET_END_ET, "%H:%M").time()

            session_start_et = pd.Timestamp(
                day.replace(hour=pm_start.hour, minute=pm_start.minute),
                tz='US/Eastern'
            )
            premarket_end_et = pd.Timestamp(
                day.replace(hour=pm_end.hour, minute=pm_end.minute),
                tz='US/Eastern'
            )
            session_end_et = pd.Timestamp(day.replace(hour=20, minute=0), tz='US/Eastern')
            warmup = session_cfg.PREMARKET_WARMUP_MINUTES
            min_bars = session_cfg.PREMARKET_MIN_BARS
        else:
            after_hours_end = datetime.strptime(session_cfg.AFTER_HOURS_END_ET, "%H:%M").time()
            session_end_et = pd.Timestamp(
                day.replace(hour=after_hours_end.hour, minute=after_hours_end.minute), tz='US/Eastern'
            )
            session_start_et = pd.Timestamp(
                day.replace(hour=9, minute=30),
                tz='US/Eastern'
            )
            premarket_end_et = session_start_et
            warmup = session_cfg.REGULAR_WARMUP_MINUTES
            min_bars = session_cfg.REGULAR_MIN_BARS

        # session_end_et is now set in both branches above — no unconditional override
        session_start_utc = session_start_et.tz_convert('UTC')
        premarket_end_utc = premarket_end_et.tz_convert('UTC')
        session_end_utc = session_end_et.tz_convert('UTC')

        # Define analysis time window (for faster backtesting)
        if self.config.backtest.ANALYSIS_WINDOW_ENABLED and not is_live:
            window_start_str = self.config.backtest.ANALYSIS_WINDOW_START_ET
            window_end_str   = self.config.backtest.ANALYSIS_WINDOW_END_ET

            window_start = datetime.strptime(window_start_str, "%H:%M").time()
            window_end   = datetime.strptime(window_end_str,   "%H:%M").time()

            analysis_window_start_et = pd.Timestamp(
                day.replace(hour=window_start.hour, minute=window_start.minute),
                tz='US/Eastern'
            )
            analysis_window_end_et = pd.Timestamp(
                day.replace(hour=window_end.hour, minute=window_end.minute),
                tz='US/Eastern'
            )
            analysis_window_start_utc = analysis_window_start_et.tz_convert('UTC')
            analysis_window_end_utc   = analysis_window_end_et.tz_convert('UTC')

            self.logger.info(
                "[ANALYSIS WINDOW] Enabled: %s - %s ET (candidates outside window will be skipped)",
                window_start_str, window_end_str
            )
        else:
            analysis_window_start_utc = None
            analysis_window_end_utc   = None

        # Check market context BEFORE analyzing candidates
        if self.market_context:
            if not self.is_live:
                self.market_context.update_market_context(day)

            if hasattr(self.market_context, 'should_trade'):
                if not self.market_context.should_trade():
                    self.logger.warning(
                        "[MARKET CONTEXT] Trading blocked: score=%.1f, environment=%s",
                        self.market_context.market_indicators.get('market_score', 0),
                        self.market_context.market_indicators.get('trading_environment', 'unknown')
                    )
                    return {"signals": signals, "diagnostics": diagnostics, "stats": stats}

            mc_score = self.market_context.market_indicators.get('market_score', 50)
            mc_env = self.market_context.market_indicators.get('trading_environment', 'neutral')
            self.logger.info("[MARKET CONTEXT] Score: %.1f | Environment: %s", mc_score, mc_env)

        min_price = self.config.screening.MIN_PRICE

        # Analyze each candidate
        for idx, (_, row) in enumerate(candidates_df.iterrows(), 1):
            symbol = str(row['symbol'])
            gap_pct = float(row['gap_percent'])
            rel_vol = row.get('relative_volume', None)
            daily_volume = row.get('daily_volume', None)
            stats["processed"] += 1

            # Optional news gate
            news_probe_ts = session_start_utc + pd.Timedelta(minutes=warmup)
            if self.news and not self._check_news_catalyst(
                symbol, day, gap_pct, rel_vol, news_probe_ts, diagnostics, stats
            ):
                continue

            # Get intraday bars
            bars = self.data.get_intraday_bars(
                symbol,
                start=session_start_utc,
                end=session_end_utc if not is_live else None
            )

            if bars is None or bars.empty:   # NOTE: add `bars is None` guard
                self.logger.debug("No intraday bars for %s, skipping", symbol)
                continue

            # Warmup check — minimum bars required for pattern analysis
            if len(bars) < warmup:
                diagnostics.append({
                    "date": pd.Timestamp(day).date(),
                    "symbol": symbol,
                    "gap_percent": gap_pct,
                    "relative_volume": rel_vol,
                    "daily_volume": daily_volume,
                    "phase": "reject",
                    "reason": "insufficient_bars_for_pattern",
                    "rows": len(bars),
                    "warmup": warmup
                })
                stats["reject_reasons"]["insufficient_bars_for_pattern"] += 1
                continue

            # Check for NaN in warmup period (pattern analysis window)
            if bars.iloc[:warmup].isna().any().any():
                diagnostics.append({
                    "date": pd.Timestamp(day).date(),
                    "symbol": symbol,
                    "gap_percent": gap_pct,
                    "relative_volume": rel_vol,
                    "daily_volume": daily_volume,
                    "phase": "reject",
                    "reason": "nan_in_pattern_window"
                })
                stats["reject_reasons"]["nan_in_pattern_window"] += 1
                continue

            # Guard: no fill bar exists beyond the warmup window at all
            if len(bars) <= warmup:
                diagnostics.append({
                    "date": pd.Timestamp(day).date(),
                    "symbol": symbol,
                    "gap_percent": gap_pct,
                    "relative_volume": rel_vol,
                    "daily_volume": daily_volume,
                    "phase": "reject",
                    "reason": "no_next_bar_for_entry",
                    "rows": len(bars),
                    "warmup": warmup
                })
                stats["reject_reasons"]["no_next_bar_for_entry"] += 1
                continue

            # Last bar pattern analysis runs on (close = signal); kept for meta audit
            signal_row = bars.iloc[warmup - 1]

            # ── Pattern analysis ─────────────────────────────────────────────────
            bars_warm = bars.iloc[:warmup].copy()

            pa = self.pa.analyze_pattern(
                symbol,
                bars=bars_warm,
                is_premarket=premarket_enabled,
                gap_percent=gap_pct
            )

            if not pa.get('valid', False):
                diagnostic = self.diagnostic_creator.create_pattern_rejection(
                    date=day,
                    symbol=symbol,
                    gap_percent=gap_pct,
                    relative_volume=rel_vol,
                    pattern_data=pa
                )
                diagnostic['daily_volume'] = daily_volume
                diagnostics.append(diagnostic)
                stats["reject_reasons"]["pattern_invalid"] += 1
                continue

            # ── Resolve fill bar ─────────────────────────────────────────────────
            # first_signal_bar_idx = confirmation bar (pivot proved by next bar closing lower)
            # fill bar             = confirmation bar + 1 (its open is the first tradeable price)
            # Falls back to warmup bar if early signal not available or no bar follows it.
            fsb_idx = pa.get('first_signal_bar_idx', -1)
            if fsb_idx >= 0 and fsb_idx + 1 < len(bars):
                fill_row = bars.iloc[fsb_idx + 1]
            elif warmup < len(bars):
                fill_row = bars.iloc[warmup]
            else:
                diagnostics.append({
                    "date": pd.Timestamp(day).date(),
                    "symbol": symbol,
                    "gap_percent": gap_pct,
                    "relative_volume": rel_vol,
                    "daily_volume": daily_volume,
                    "phase": "reject",
                    "reason": "no_next_bar_for_entry",
                    "rows": len(bars),
                    "warmup": warmup
                })
                stats["reject_reasons"]["no_next_bar_for_entry"] += 1
                continue

            entry_price = float(fill_row['open'])
            entry_ts    = fill_row['timestamp']

            # ── Price floor check on the resolved fill bar (single, canonical check) ──
            if entry_price < min_price:
                diagnostics.append({
                    "date": pd.Timestamp(day).date(),
                    "symbol": symbol,
                    "gap_percent": gap_pct,
                    "relative_volume": rel_vol,
                    "daily_volume": daily_volume,
                    "phase": "reject",
                    "reason": "price_out_of_range",
                    "entry_price": entry_price,
                    "min_price": min_price
                })
                stats["reject_reasons"]["price_out_of_range"] += 1
                stats["after_price_filter"] = stats["processed"] - stats["reject_reasons"]["price_out_of_range"]
                self.logger.debug(
                    "[REJECT] %s - Price $%.2f below minimum ($%.2f)",
                    symbol, entry_price, min_price
                )
                continue

            # ── Analysis window gate on resolved fill bar timestamp ───────────────
            if analysis_window_start_utc is not None and analysis_window_end_utc is not None:
                if isinstance(entry_ts, pd.Timestamp):
                    entry_ts_utc = entry_ts.tz_localize('UTC') if entry_ts.tz is None else entry_ts.tz_convert('UTC')
                else:
                    entry_ts_utc = pd.Timestamp(entry_ts).tz_localize('UTC')

                if entry_ts_utc < analysis_window_start_utc or entry_ts_utc >= analysis_window_end_utc:
                    entry_et = entry_ts_utc.tz_convert('US/Eastern')
                    diagnostics.append({
                        "date": pd.Timestamp(day).date(),
                        "symbol": symbol,
                        "gap_percent": gap_pct,
                        "relative_volume": rel_vol,
                        "daily_volume": daily_volume,
                        "phase": "reject",
                        "reason": "outside_analysis_window",
                        "entry_time_et": entry_et.strftime('%H:%M:%S'),
                        "window_start_et": window_start_str,
                        "window_end_et": window_end_str
                    })
                    stats["reject_reasons"]["outside_analysis_window"] += 1
                    self.logger.debug(
                        "[SKIP] %s - Entry time %s ET outside analysis window (%s - %s)",
                        symbol, entry_et.strftime('%H:%M:%S'), window_start_str, window_end_str
                    )
                    continue

            # ── Valid signal ──────────────────────────────────────────────────────
            stats["pattern_valid"] += 1

            is_premarket_entry = premarket_enabled and entry_ts < premarket_end_utc

            stop_price = calc_atr_stop(
                bars_warm,
                entry_price,
                atr_period=14,
                atr_mult=2.0,
                fallback_pct=0.03
            )

            self.logger.info(
                "[SIGNAL] %s | Gap: %+.2f%% | Entry: $%.2f | Stop: $%.2f | "
                "Pattern Score: %.1f | Daily Vol: %s | Premarket: %s",
                symbol, gap_pct, entry_price, stop_price,
                pa.get('pattern_strength', 0),
                f"{daily_volume:,.0f}" if daily_volume else "N/A",
                is_premarket_entry
            )

            signals.append(CandidateSignal(
                symbol=symbol,
                entry_ts=entry_ts,
                entry_price=entry_price,
                stop_price=stop_price,
                gap_percent=gap_pct,
                pattern_strength=float(pa.get('pattern_strength', 0) or 0),
                relative_volume=rel_vol,
                meta={
                    "last_price":       float(row['last_price']),
                    "open_price":       float(row.get('open_price', entry_price)),
                    "prev_close":       float(row.get('prev_close', entry_price / (1 + gap_pct/100))),
                    "is_premarket":     is_premarket_entry,
                    "daily_volume":     int(daily_volume) if daily_volume else None,
                    "signal_bar_ts":    signal_row['timestamp'],
                    "signal_bar_close": float(signal_row['close']),
                }
            ))
            stats["signals"] += 1

        # Final price filter stat (if not already set by rejections)
        if "price_out_of_range" not in stats["reject_reasons"]:
            stats["after_price_filter"] = stats["processed"]

        return {"signals": signals, "diagnostics": diagnostics, "stats": stats}

    def _apply_daily_volume_filter(
        self,
        candidates_df: pd.DataFrame,
        day: datetime,
        stats: Dict,
        day_aggregates: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply daily volume pre-screen using aggregate data.

        This checks total daily volume BEFORE loading intraday data,
        which is much faster than loading bars for every candidate.
        """
        min_daily_vol = self.config.screening.MIN_DAILY_VOLUME

        self.logger.info(
            "[DAILY VOLUME PRE-SCREEN] Filtering %d candidates (min: %d)",
            len(candidates_df), min_daily_vol
        )

        if day_aggregates is None or day_aggregates.empty:
            self.logger.warning("No aggregate data available for %s - skipping volume filter", day.date())
            stats["after_daily_volume_filter"] = len(candidates_df)
            return candidates_df

        # Create volume lookup
        volume_lookup = day_aggregates.set_index('symbol')['volume'].to_dict()

        # Add daily volume to candidates
        candidates_df['daily_volume'] = candidates_df['symbol'].map(volume_lookup)

        # Filter by minimum daily volume
        pre_filter = len(candidates_df)
        candidates_df = candidates_df[
            (candidates_df['daily_volume'].notna()) &
            (candidates_df['daily_volume'] >= min_daily_vol)
        ]
        stats["after_daily_volume_filter"] = len(candidates_df)

        filtered_count = pre_filter - len(candidates_df)

        self.logger.info(
            "[FILTER] Daily volume pre-screen | Before: %d | After: %d | Filtered: %d",
            pre_filter, len(candidates_df), filtered_count
        )

        if filtered_count > 0:
            stats["reject_reasons"]["low_daily_volume"] = filtered_count

        return candidates_df

    def _apply_rvol_filter(
        self,
        candidates_df: pd.DataFrame,
        day: datetime,
        is_live: bool,
        stats: Dict
    ) -> pd.DataFrame:
        """Apply relative volume filter."""
        max_rvol_checks = min(
            len(candidates_df),
            self.config.backtest.MAX_CANDIDATES_PER_DAY * 2
        )
        top_candidates = candidates_df.head(max_rvol_checks)

        self.logger.info(
            "[RELATIVE VOLUME] Calculating for top %d candidates",
            len(top_candidates)
        )

        symbols_to_check = top_candidates['symbol'].tolist()

        # NOTE: Use appropriate calculator based on mode
        if is_live:
            # For live: calculate_batch doesn't need day parameter
            rel_vols = self.rvol_calculator.calculate_batch(symbols_to_check)
        else:
            # For backtest: calculate_batch needs day parameter
            rel_vols = self.rvol_calculator.calculate_batch(symbols_to_check, day)

        candidates_df['relative_volume'] = candidates_df['symbol'].map(rel_vols).fillna(0.0)

        min_rel_vol = self.config.screening.MIN_RELATIVE_VOLUME
        pre_relvol = len(candidates_df)
        candidates_df = candidates_df[candidates_df['relative_volume'] >= min_rel_vol]
        stats["after_relative_volume_filter"] = len(candidates_df)

        filtered_by_relvol = pre_relvol - len(candidates_df)
        self.logger.info(
            "[FILTER] Applied relative volume filter (>=%fx) | Before: %d | After: %d | Filtered: %d",
            min_rel_vol, pre_relvol, len(candidates_df), filtered_by_relvol
        )

        if filtered_by_relvol > 0:
            stats["reject_reasons"]["low_relative_volume"] = filtered_by_relvol

        return candidates_df

    def _apply_fundamental_filters(
        self,
        candidates_df: pd.DataFrame,
        day: datetime,
        stats: Dict,
        day_aggregates: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply fundamental filters (float and market cap) using aggregate data.

        This filters stocks based on shares outstanding and market capitalization
        BEFORE loading intraday data.
        """
        enable_float = self.config.screening.ENABLE_FLOAT_FILTER
        enable_marketcap = self.config.screening.ENABLE_MARKETCAP_FILTER

        if not enable_float and not enable_marketcap:
            stats["after_fundamental_filter"] = len(candidates_df)
            return candidates_df

        max_float = self.config.screening.MAX_FLOAT
        max_marketcap = self.config.screening.MAX_MARKETCAP

        self.logger.info(
            "[FUNDAMENTAL FILTER] Filtering %d candidates (max float: %d, max marketcap: $%d)",
            len(candidates_df), max_float, max_marketcap
        )

        if day_aggregates is None or day_aggregates.empty:
            self.logger.warning("No aggregate data for fundamental filter on %s", day.date())
            stats["after_fundamental_filter"] = len(candidates_df)
            return candidates_df

        # Check if fundamental columns exist
        has_float = 'float' in day_aggregates.columns
        has_marketcap = 'marketcap' in day_aggregates.columns

        if not has_float and not has_marketcap:
            self.logger.warning("No fundamental data (float/marketcap) in aggregates - skipping filter")
            stats["after_fundamental_filter"] = len(candidates_df)
            return candidates_df

        # Create lookup dictionaries
        if has_float:
            float_lookup = day_aggregates.set_index('symbol')['float'].to_dict()
        if has_marketcap:
            marketcap_lookup = day_aggregates.set_index('symbol')['marketcap'].to_dict()

        # Add fundamental data to candidates
        candidates_df = candidates_df.copy()

        if has_float and enable_float:
            candidates_df['float'] = candidates_df['symbol'].map(float_lookup)

        if has_marketcap and enable_marketcap:
            candidates_df['marketcap'] = candidates_df['symbol'].map(marketcap_lookup)

        # Apply filters
        pre_filter = len(candidates_df)
        filter_mask = pd.Series([True] * len(candidates_df), index=candidates_df.index)

        filtered_reasons = {}

        if enable_float and has_float:
            float_mask = (
                candidates_df['float'].notna() &
                (candidates_df['float'] <= max_float)
            )
            failed_float = (~float_mask).sum()
            if failed_float > 0:
                filtered_reasons['high_float'] = failed_float
                self.logger.debug("  Float filter: %d stocks exceed %d shares", failed_float, max_float)
            filter_mask &= float_mask

        if enable_marketcap and has_marketcap:
            marketcap_mask = (
                candidates_df['marketcap'].notna() &
                (candidates_df['marketcap'] <= max_marketcap)
            )
            failed_marketcap = (~marketcap_mask).sum()
            if failed_marketcap > 0:
                filtered_reasons['high_marketcap'] = failed_marketcap
                self.logger.debug("  Marketcap filter: %d stocks exceed $%d", failed_marketcap, max_marketcap)
            filter_mask &= marketcap_mask

        candidates_df = candidates_df[filter_mask]
        stats["after_fundamental_filter"] = len(candidates_df)

        filtered_count = pre_filter - len(candidates_df)

        self.logger.info(
            "[FILTER] Fundamental filter | Before: %d | After: %d | Filtered: %d",
            pre_filter, len(candidates_df), filtered_count
        )

        if filtered_count > 0:
            for reason, count in filtered_reasons.items():
                stats["reject_reasons"][reason] = count

        return candidates_df

    def _check_news_catalyst(
        self,
        symbol: str,
        day: datetime,
        gap_pct: float,
        rel_vol: Optional[float],
        entry_time: pd.Timestamp,  # NOTE: NEW: Required for time-aware checking
        diagnostics: List[Dict],
        stats: Dict
    ) -> bool:
        """
        Check news catalyst with time-aware filtering.

        Rules:
        1. If neg > 0.08, reject immediately
        2. If neg <= 0.08, only allow trading after news published
        """
        ignore_catalyst = getattr(self.config.backtest, "IGNORE_CATALYST", True)

        if ignore_catalyst:
            return True

        try:
            # NOTE: Use time-aware checking
            news = self.news.check_news_approval(symbol, entry_time)
        except Exception as e:
            self.logger.debug("[NEWS] %s - Analysis error: %s", symbol, e)
            news = {'approved': False, 'reason': 'error'}

        # Check approval
        if not news.get('approved', False):
            reason = news.get('reason', 'unknown')

            diagnostics.append(self.diagnostic_creator.create_rejection(
                date=day,
                symbol=symbol,
                gap_percent=gap_pct,
                relative_volume=rel_vol,
                reason=f"news_{reason}",
                max_negative=news.get('max_negative', 0.0),
                earliest_news_time=news.get('earliest_news_time'),
            ))
            stats["reject_reasons"][f"news_{reason}"] += 1

            if reason == 'negative_sentiment':
                self.logger.debug(
                    "[REJECT] %s - Negative sentiment: %.3f > %.3f",
                    symbol, news.get('max_negative', 0), getattr(self.config.backtest, 'MAX_NEGATIVE_SENTIMENT', 0.08)
                )
            elif reason == 'insufficient_positive_sentiment':
                self.logger.debug(
                    "[REJECT] %s - Insufficient positive sentiment",
                    symbol
                )
            elif reason == 'news_not_yet_published':
                earliest = news.get('earliest_news_time')
                if earliest:
                    earliest_et = earliest.tz_convert('US/Eastern')
                    entry_et = entry_time.tz_convert('US/Eastern')
                    self.logger.debug(
                        "[REJECT] %s - News not yet published at entry time "
                        "(entry: %s ET, earliest news: %s ET)",
                        symbol, entry_et.strftime('%H:%M:%S'), earliest_et.strftime('%H:%M:%S')
                    )
            else:
                self.logger.debug(
                    "[REJECT] %s - %s",
                    symbol, reason
                )

            return False

        # PASSED: sentiment gates cleared and news already published
        max_neg = news.get('max_negative', 0.0)
        earliest = news.get('earliest_news_time')

        if earliest:
            earliest_et = earliest.tz_convert('US/Eastern')
            entry_et = entry_time.tz_convert('US/Eastern')
            time_diff = (entry_time - earliest).total_seconds() / 60

            self.logger.info(
                "[NEWS] %s - PASS: max_neg=%.3f, earliest=%s ET, entry=%s ET (%.0fmin after news)",
                symbol, max_neg, earliest_et.strftime('%H:%M'), entry_et.strftime('%H:%M'), time_diff
            )
        return True