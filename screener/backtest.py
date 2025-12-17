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
from core.pattern_analyzer import PatternAnalyzer
from news.backtest import NewsIntegrationBacktest
from screener.core import UnifiedScreener, CandidateSignal


class BacktestScreener:
    """Backtest screener using unified logic."""
    
    def __init__(self, config, data_handler, pattern_analyzer: PatternAnalyzer, 
                 news_integration: Optional[NewsIntegrationBacktest] = None):
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
        self.logger.info(f"SCREENING DAY: {day.date()}")
        self.logger.info("=" * 80)
        
        # ✅ FIX: Use AggregateDataHandler directly instead of deprecated calculate_gaps()
        from data_handler.aggregate_handler import AggregateDataHandler
        
        agg_dir = Path(self.config.backtest.BASE_DATA_DIR) / "daily_aggregates"
        agg_handler = AggregateDataHandler(str(agg_dir))
        
        # Get current day aggregates
        day_agg = agg_handler.get_day_aggregates(day)
        
        if day_agg is None or day_agg.empty:
            self.logger.warning(f"No aggregate data for {day.date()}")
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
            self.logger.warning(f"No previous day data found within 7 days")
            return {"signals": [], "diagnostics": [], "stats": {}}
        
        # Calculate gaps with validation
        common_symbols = set(day_agg['symbol']) & set(prev_agg['symbol'])
        
        gaps_list = []
        for symbol in common_symbols:
            day_row = day_agg[day_agg['symbol'] == symbol].iloc[0]
            prev_row = prev_agg[prev_agg['symbol'] == symbol].iloc[0]
            
            # ✅ FIX: Validate prev_close before division
            prev_close = prev_row['close']
            if pd.isna(prev_close) or prev_close <= 0:
                self.logger.debug(f"Skipping {symbol}: invalid prev_close={prev_close}")
                continue
            
            # ✅ FIX: Validate day open price
            day_open = day_row['open']
            if pd.isna(day_open) or day_open <= 0:
                self.logger.debug(f"Skipping {symbol}: invalid open={day_open}")
                continue
            
            gap_pct = ((day_open - prev_close) / prev_close) * 100.0
            
            # ✅ FIX: Skip if gap calculation resulted in invalid value
            if not np.isfinite(gap_pct):
                self.logger.debug(f"Skipping {symbol}: invalid gap_pct={gap_pct}")
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
            self.logger.warning(f"No gaps calculated for {day.date()}")
            return {"signals": [], "diagnostics": [], "stats": {}}
        
        self.logger.info(f"Calculated {len(gaps_df)} gaps (using {prev_day.strftime('%Y-%m-%d')} as prev day)")
        
        # Apply basic filters
        min_gap = self.config.screening.MIN_GAP_PERCENT
        candidates = gaps_df[gaps_df['gap_percent'] >= min_gap]
        
        if candidates.empty:
            self.logger.warning(f"No candidates meeting gap threshold ({min_gap}%)")
            return {"signals": [], "diagnostics": [], "stats": {}}
        
        self.logger.info(f"Candidates after gap filter: {len(candidates)}")
        
        # Use unified screener logic
        result = self.core_screener.screen_symbols(
            candidates_df=candidates,
            day=day,
            is_live=False
        )
        
        # Log summary
        self.logger.info("=" * 80)
        self.logger.info(f"DAY COMPLETE: {day.date()}")
        self.logger.info(f"  • Signals Generated: {result['stats']['signals']}")
        self.logger.info(f"  • Pattern Valid: {result['stats']['pattern_valid']}")
        self.logger.info("=" * 80)
        
        return result