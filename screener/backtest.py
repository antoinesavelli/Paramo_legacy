# =====================================================
# screener.backtest.py - Backtest screener wrapper
# =====================================================

from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from utils.logging import get_logger
from utils.helpers import calc_gap_percent
from strategy.patterns.pattern_analyzer import PatternAnalyzer
from news.backtest import NewsIntegrationBacktest
from screener.core import UnifiedScreener, CandidateSignal
from data_handler.base import DataHandler  # ADD


class BacktestScreener:
    """Backtest screener using unified logic."""
    
    def __init__(
        self,
        config,
        data_handler: DataHandler,          # ← was untyped
        pattern_analyzer: PatternAnalyzer,
        news_integration: Optional[NewsIntegrationBacktest] = None,
    ):
        self.config = config
        self.data = data_handler
        self.logger = get_logger(__name__, component="bt_screener")
        
        # Use unified screener with is_live=False
        self.core_screener = UnifiedScreener(
            config,
            data_handler,
            pattern_analyzer,
            news_integration=news_integration,
            logger=self.logger,
            is_live=False
        )
    
    def screen(self, day: datetime) -> Dict[str, List]:
        """Screen for candidates on a specific historical day."""
        
        # Check data availability
        day_str = day.strftime('%Y-%m-%d')
        if not self.data.has_data_for_date(day_str):
            return {"signals": [], "diagnostics": [], "stats": {}}
        
        self.logger.info("=" * 80)
        self.logger.info("SCREENING DAY: %s", day.date())
        self.logger.info("=" * 80)
        
        # NOTE: FIX: Use AggregateDataHandler directly instead of deprecated calculate_gaps()
        from data_handler.aggregates.aggregate_handler import AggregateDataHandler
        
        agg_dir = Path(self.config.backtest.BASE_DATA_DIR) / "daily_aggregates"
        agg_handler = AggregateDataHandler(str(agg_dir))
        
        # Get current day aggregates
        day_agg = agg_handler.get_day_aggregates(day)
        
        if day_agg is None or day_agg.empty:
            self.logger.warning("No aggregate data for %s", day.date())
            return {"signals": [], "diagnostics": [], "stats": {}}
        
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
            self.logger.warning("No previous day data found within 7 days")
            return {"signals": [], "diagnostics": [], "stats": {}}
        
        # Calculate gaps with validation
        # Pre-index by symbol for O(1) lookup (M4: avoids O(n²) filter-in-loop)
        day_idx = day_agg.set_index('symbol')
        prev_idx = prev_agg.set_index('symbol')
        common_symbols = set(day_idx.index) & set(prev_idx.index)

        # Filter out symbols with a split effective on this day (H6: corporate actions)
        if self.config.screening.FILTER_SPLIT_DAYS:
            split_symbols = self._get_split_symbols_for_day(day)
            if split_symbols:
                filtered = common_symbols - split_symbols
                self.logger.info(
                    "Filtered %d split-day symbol(s) from %d candidates: %s",
                    len(split_symbols & common_symbols),
                    len(common_symbols),
                    sorted(split_symbols & common_symbols),
                )
                common_symbols = filtered

        gaps_list = []
        for symbol in common_symbols:
            day_row = day_idx.loc[symbol]
            prev_row = prev_idx.loc[symbol]
            
            # NOTE: FIX: Validate prev_close before division
            prev_close = prev_row['close']
            if pd.isna(prev_close) or prev_close <= 0:
                self.logger.debug("Skipping %s: invalid prev_close=%s", symbol, prev_close)
                continue
            
            # NOTE: FIX: Validate day open price
            day_open = day_row['open']
            if pd.isna(day_open) or day_open <= 0:
                self.logger.debug("Skipping %s: invalid open=%s", symbol, day_open)
                continue
            
            gap_pct = calc_gap_percent(day_open, prev_close)
            
            # NOTE: FIX: Skip if gap calculation resulted in invalid value
            if not np.isfinite(gap_pct):
                self.logger.debug("Skipping %s: invalid gap_pct=%s", symbol, gap_pct)
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
        
        if gaps_df.empty:
            self.logger.warning("No gaps calculated for %s", day.date())
            return {"signals": [], "diagnostics": [], "stats": {}}
        
        self.logger.info("Calculated %d gaps (using %s as prev day)", len(gaps_df), prev_day.strftime('%Y-%m-%d'))
        
        # Apply basic filters
        min_gap = self.config.screening.MIN_GAP_PERCENT
        candidates = gaps_df[gaps_df['gap_percent'] >= min_gap]
        
        if candidates.empty:
            self.logger.warning("No candidates meeting gap threshold (%.1f%%)", min_gap)
            return {"signals": [], "diagnostics": [], "stats": {}}
        
        self.logger.info("Candidates after gap filter: %d", len(candidates))
        
        # Use unified screener logic
        result = self.core_screener.screen_symbols(
            candidates_df=candidates,
            day=day,
            is_live=False
        )
        
        # Log summary
        self.logger.info("=" * 80)
        self.logger.info("DAY COMPLETE: %s", day.date())
        self.logger.info("  • Signals Generated: %s", result['stats']['signals'])
        self.logger.info("  • Pattern Valid: %s", result['stats']['pattern_valid'])
        self.logger.info("=" * 80)
        
        return result

    def _get_split_symbols_for_day(self, day: datetime) -> set:
        """
        Return symbols with a split effective on *day* by querying Alpaca's
        corporate actions endpoint.

        Requires valid ALPACA_API_KEY / ALPACA_SECRET_KEY in config.api.
        Returns an empty set silently if keys are absent or on any error —
        the screener continues unaffected.
        """
        api_key = self.config.api.ALPACA_API_KEY
        secret_key = self.config.api.ALPACA_SECRET_KEY
        base_url = self.config.api.ALPACA_BASE_URL
        if not api_key or not secret_key:
            return set()
        try:
            import alpaca_trade_api as tradeapi
            from utils.helpers import fetch_split_symbols
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            return fetch_split_symbols(api, day)
        except Exception as exc:
            self.logger.debug("Could not fetch split data for %s: %s", day.date(), exc)
            return set()