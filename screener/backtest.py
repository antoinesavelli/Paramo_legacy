# =====================================================
# screener.backtest.py - Backtest screener wrapper
# =====================================================

from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

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
            is_live=False  # This tells it to use BacktestRelativeVolumeCalculator
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
        
        # Calculate gaps
        gaps_info = self.data.calculate_gaps(day)  # ✅ Remove premarket parameter
        gaps_df: pd.DataFrame = gaps_info.get('gaps', pd.DataFrame())
        
        if gaps_df.empty:
            self.logger.warning(f"No gaps found for {day.date()}")
            return {"signals": [], "diagnostics": [], "stats": {}}
        
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