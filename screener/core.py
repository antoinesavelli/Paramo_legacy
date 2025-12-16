# =====================================================
# screener.core.py - Unified screening logic (live + backtest)
# =====================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import collections

from utils.logging import get_logger
from core.risk_manager import calc_atr_stop
from core.pattern_analyzer import PatternAnalyzer
from screener.rules import cfg_view_from
from screener.helpers import (
    LiveRelativeVolumeCalculator, 
    BacktestRelativeVolumeCalculator, 
    DiagnosticCreator
)


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
        market_context=None,  # ✅ NEW: Add market context
        logger=None, 
        is_live=False
    ):
        self.config = config
        self.data = data_handler
        self.pa = pattern_analyzer
        self.news = news_integration
        self.market_context = market_context  # ✅ NEW
        self.logger = logger or get_logger(__name__, component="screener")
        self.is_live = is_live  # ✅ NEW: Store mode
        
        # ✅ Initialize appropriate RVOL calculator based on mode
        if is_live:
            self.rvol_calculator = LiveRelativeVolumeCalculator(config, data_handler, self.logger)
        else:
            self.rvol_calculator = BacktestRelativeVolumeCalculator(config, data_handler, self.logger)
        
        self.diagnostic_creator = DiagnosticCreator(config)
    
    def screen_symbols(
        self, 
        candidates_df: pd.DataFrame,
        day: datetime,
        is_live: bool = False
    ) -> Dict[str, List]:
        """
        Unified screening logic for both live and backtest.
        
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
            "after_relative_volume_filter": 0,
            "processed": 0,
            "pattern_valid": 0,
            "signals": 0,
            "reject_reasons": collections.Counter()
        }
        
        if candidates_df.empty:
            return {"signals": signals, "diagnostics": diagnostics, "stats": stats}
        
        # Sort by gap % (highest first)
        candidates_df = candidates_df.sort_values('gap_percent', ascending=False)
        
        # Apply RVOL filter if enabled
        if self.config.screening.ENABLE_RELATIVE_VOLUME:
            candidates_df = self._apply_rvol_filter(
                candidates_df, day, is_live, stats
            )
            
            if candidates_df.empty:
                self.logger.warning("All candidates filtered by relative volume")
                return {"signals": signals, "diagnostics": diagnostics, "stats": stats}
        else:
            stats["after_relative_volume_filter"] = len(candidates_df)
        
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
            warmup = session_cfg.PREMARKET_WARMUP_MINUTES
            min_bars = session_cfg.PREMARKET_MIN_BARS
        else:
            session_start_et = pd.Timestamp(
                day.replace(hour=9, minute=30), 
                tz='US/Eastern'
            )
            premarket_end_et = session_start_et
            warmup = session_cfg.REGULAR_WARMUP_MINUTES
            min_bars = session_cfg.REGULAR_MIN_BARS
        
        session_end_et = pd.Timestamp(day.replace(hour=20, minute=0), tz='US/Eastern')
        session_start_utc = session_start_et.tz_convert('UTC')
        premarket_end_utc = premarket_end_et.tz_convert('UTC')
        session_end_utc = session_end_et.tz_convert('UTC')
        
        # ✅ NEW: Check market context BEFORE analyzing candidates
        if self.market_context:
            # Update context (for backtest, pass day; for live, it updates from API)
            if not self.is_live:
                self.market_context.update_market_context(day)
            
            # Check if we should trade today
            if hasattr(self.market_context, 'should_trade'):
                if not self.market_context.should_trade():
                    self.logger.warning(
                        f"[MARKET CONTEXT] Trading blocked: "
                        f"score={self.market_context.market_indicators.get('market_score', 0):.1f}, "
                        f"environment={self.market_context.market_indicators.get('trading_environment', 'unknown')}"
                    )
                    return {"signals": signals, "diagnostics": diagnostics, "stats": stats}
        
            # Log market context
            mc_score = self.market_context.market_indicators.get('market_score', 50)
            mc_env = self.market_context.market_indicators.get('trading_environment', 'neutral')
            self.logger.info(
                f"[MARKET CONTEXT] Score: {mc_score:.1f} | Environment: {mc_env}"
            )
        
        # Analyze each candidate
        for idx, (_, row) in enumerate(candidates_df.iterrows(), 1):
            symbol = str(row['symbol'])
            gap_pct = float(row['gap_percent'])
            rel_vol = row.get('relative_volume', None)
            stats["processed"] += 1
            
            # Optional news gate
            if self.news and not self._check_news_catalyst(
                symbol, day, gap_pct, rel_vol, diagnostics, stats
            ):
                continue
            
            # Get intraday bars
            bars = self.data.get_intraday_bars(
                symbol, 
                start=session_start_utc, 
                end=session_end_utc if not is_live else None
            )
            
            if bars.empty:
                diagnostics.append(self.diagnostic_creator.create_rejection(
                    date=day,
                    symbol=symbol,
                    gap_percent=gap_pct,
                    relative_volume=rel_vol,
                    reason="empty_bars"
                ))
                stats["reject_reasons"]["empty_bars"] += 1
                continue
            
            # Bar count checks
            if premarket_enabled and len(bars) < min_bars:
                diagnostics.append({
                    "date": pd.Timestamp(day).date(),
                    "symbol": symbol,
                    "gap_percent": gap_pct,
                    "relative_volume": rel_vol,
                    "phase": "reject",
                    "reason": "premarket_min_bars",
                    "rows": len(bars),
                    "min_bars": min_bars
                })
                stats["reject_reasons"]["premarket_min_bars"] += 1
                continue
            
            if len(bars) < warmup:
                diagnostics.append({
                    "date": pd.Timestamp(day).date(),
                    "symbol": symbol,
                    "gap_percent": gap_pct,
                    "relative_volume": rel_vol,
                    "phase": "reject",
                    "reason": "warmup_short",
                    "rows": len(bars),
                    "warmup": warmup
                })
                stats["reject_reasons"]["warmup_short"] += 1
                continue
            
            # Validate columns
            missing_cols = [
                col for col in ['open', 'high', 'low', 'close', 'volume', 'timestamp'] 
                if col not in bars.columns
            ]
            if missing_cols:
                diagnostics.append({
                    "date": pd.Timestamp(day).date(),
                    "symbol": symbol,
                    "gap_percent": gap_pct,
                    "relative_volume": rel_vol,
                    "phase": "reject",
                    "reason": "bars_missing_columns",
                    "missing": ",".join(missing_cols)
                })
                stats["reject_reasons"]["bars_missing_columns"] += 1
                continue
            
            # Check for NaN in warmup period
            if bars.iloc[:warmup].isna().any().any():
                diagnostics.append({
                    "date": pd.Timestamp(day).date(),
                    "symbol": symbol,
                    "gap_percent": gap_pct,
                    "relative_volume": rel_vol,
                    "phase": "reject",
                    "reason": "nan_in_warmup"
                })
                stats["reject_reasons"]["nan_in_warmup"] += 1
                continue
            
            # Pattern analysis
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
                diagnostics.append(diagnostic)
                stats["reject_reasons"]["pattern_invalid"] += 1
                continue
            
            # Valid signal - calculate entry and stop
            stats["pattern_valid"] += 1
            entry_row = bars.iloc[warmup - 1]
            entry_price = float(entry_row['close'])
            entry_ts = entry_row['timestamp']
            
            is_premarket_entry = premarket_enabled and entry_ts < premarket_end_utc
            
            stop_price = calc_atr_stop(
                bars_warm, 
                entry_price, 
                atr_period=14, 
                atr_mult=2.0, 
                fallback_pct=0.03
            )
            
            self.logger.info(
                f"[SIGNAL] {symbol} | "
                f"Gap: {gap_pct:+.2f}% | "
                f"Entry: ${entry_price:.2f} | "
                f"Stop: ${stop_price:.2f} | "
                f"Pattern Score: {pa.get('pattern_strength', 0):.1f} | "
                f"Premarket: {is_premarket_entry}"
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
                    "last_price": float(row['last_price']),
                    "open_price": float(row.get('open_price', entry_price)),
                    "prev_close": float(row.get('prev_close', entry_price / (1 + gap_pct/100))),
                    "is_premarket": is_premarket_entry
                }
            ))
            stats["signals"] += 1
        
        return {"signals": signals, "diagnostics": diagnostics, "stats": stats}
    
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
            f"[RELATIVE VOLUME] Calculating for top {len(top_candidates)} candidates"
        )
        
        symbols_to_check = top_candidates['symbol'].tolist()
        
        # ✅ Use appropriate calculator based on mode
        if is_live:
            # For live: calculate_batch doesn't need day parameter
            rel_vols = self.rvol_calculator.calculate_batch(symbols_to_check)
        else:
            # For backtest: calculate_batch needs day parameter
            rel_vols = self.rvol_calculator.calculate_batch(symbols_to_check, day)
        
        candidates_df = candidates_df.copy()
        candidates_df['relative_volume'] = candidates_df['symbol'].map(rel_vols).fillna(0.0)
        
        min_rel_vol = self.config.screening.MIN_RELATIVE_VOLUME
        pre_relvol = len(candidates_df)
        candidates_df = candidates_df[candidates_df['relative_volume'] >= min_rel_vol]
        stats["after_relative_volume_filter"] = len(candidates_df)
        
        filtered_by_relvol = pre_relvol - len(candidates_df)
        self.logger.info(
            f"[FILTER] Applied relative volume filter (>={min_rel_vol}x) | "
            f"Before: {pre_relvol} | After: {len(candidates_df)} | Filtered: {filtered_by_relvol}"
        )
        
        if filtered_by_relvol > 0:
            stats["reject_reasons"]["low_relative_volume"] = filtered_by_relvol
        
        return candidates_df
    
    def _calculate_rvol_live(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate RVOL for live trading using current quote data."""
        rel_vols = {}
        lookback = self.config.screening.RELATIVE_VOLUME_LOOKBACK_DAYS
        
        for symbol in symbols:
            try:
                # Get historical daily bars
                daily_bars = self.data.get_intraday_bars(
                    symbol, 
                    timeframe='1Day', 
                    limit=lookback
                )
                
                if daily_bars is None or daily_bars.empty:
                    continue
                
                # Get current quote
                quotes = self.data.get_quote_data([symbol]) or {}
                quote = quotes.get(symbol)
                
                if not quote:
                    continue
                
                current_volume = quote.get('volume', 0)
                
                if current_volume <= 0:
                    continue
                
                # Calculate baseline (exclude today)
                valid_vols = daily_bars[daily_bars['volume'] > 0]
                if len(valid_vols) < 5:
                    continue
                
                avg_volume = valid_vols['volume'].iloc[:-1].mean()
                
                if avg_volume <= 0:
                    continue
                
                rel_vol = current_volume / avg_volume
                rel_vols[symbol] = rel_vol
                
            except Exception as e:
                self.logger.debug(f"Error calculating live RVOL for {symbol}: {e}")
                continue
        
        return rel_vols
    
    def _check_news_catalyst(
        self, 
        symbol: str, 
        day: datetime, 
        gap_pct: float, 
        rel_vol: Optional[float],
        diagnostics: List[Dict], 
        stats: Dict
    ) -> bool:
        """Check news catalyst if news integration is enabled."""
        min_news_strength = getattr(self.config.backtest, "MIN_NEWS_STRENGTH", 30)
        ignore_catalyst = getattr(self.config.backtest, "IGNORE_CATALYST", True)
        
        if ignore_catalyst:
            return True
        
        try:
            news = self.news.analyze_news_impact(symbol, date=day)
        except Exception as e:
            self.logger.debug(f"[NEWS] {symbol} - Analysis error: {e}")
            news = {}
        
        if not news.get('has_catalyst') or int(news.get('catalyst_strength', 0)) < min_news_strength:
            diagnostics.append(self.diagnostic_creator.create_rejection(
                date=day,
                symbol=symbol,
                gap_percent=gap_pct,
                relative_volume=rel_vol,
                reason="weak_catalyst",
                catalyst_strength=news.get('catalyst_strength')
            ))
            stats["reject_reasons"]["weak_catalyst"] += 1
            return False
        
        return True