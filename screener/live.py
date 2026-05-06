# =====================================================
# screener.live.py - Live screener wrapper
# =====================================================

from __future__ import annotations
from typing import List, Dict, Union, Optional
from datetime import datetime
import pandas as pd

from config import TradingConfig
from data_handler.api import APIDataHandler
from strategy.patterns.pattern_analyzer import PatternAnalyzer
from utils import get_logger
from screener.core import UnifiedScreener


class LiveScreener:
    """Live screener using unified logic."""
    
    def __init__(self, config: TradingConfig, data_handler: APIDataHandler, 
                 pattern_analyzer: PatternAnalyzer):
        self.config = config
        self.data_handler = data_handler
        self.logger = get_logger(__name__, component="screener_live")
        
        # Use unified screener
        self.core_screener = UnifiedScreener(
            config, 
            data_handler, 
            pattern_analyzer,
            news_integration=None,  # No news in live (for now)
            logger=self.logger,
            is_live=True  # This tells it to use LiveRelativeVolumeCalculator
        )
        
        self.screened_stocks: List[Dict] = []
        self.last_screen_time: Optional[datetime] = None
    
    def run_screen(self) -> List[Dict]:
        """Run live screening."""
        self.logger.info("Starting live screen...")
        
        # Get universe
        universe = self.data_handler.get_universe()
        if universe is None or universe.empty:
            self.logger.info("Universe empty")
            return []
        
        symbols = universe["symbol"].tolist()
        self.logger.info(f"Universe loaded with {len(symbols)} symbols")
        
        # Calculate gaps
        gaps_df = self.data_handler.calculate_gaps(symbols)
        if gaps_df is None or gaps_df.empty:
            self.logger.info("No gaps computed")
            return []
        
        # Apply basic filters
        min_gap = self.config.screening.MIN_GAP_PERCENT
        candidates = gaps_df[gaps_df["gap_percent"] >= min_gap]
        
        if candidates.empty:
            self.logger.info("No stocks meeting gap criteria (min=%s%%)", min_gap)
            return []

        self.logger.info("Candidates meeting gap criteria: %d", len(candidates))

        # Use unified screener logic
        result = self.core_screener.screen_symbols(
            candidates_df=candidates,
            day=datetime.now(),
            is_live=True
        )
        
        # Convert signals to dict format
        self.screened_stocks = [
            {
                "symbol": sig.symbol,
                "entry_price": sig.entry_price,
                "stop_price": sig.stop_price,
                "gap_percent": sig.gap_percent,
                "pattern_strength": sig.pattern_strength,
                "relative_volume": sig.relative_volume,
                "entry_ts": sig.entry_ts,
                **sig.meta
            }
            for sig in result["signals"]
        ]
        
        self.last_screen_time = datetime.now()
        self.logger.info("Screening complete. Signals: %d", len(self.screened_stocks))

        return self.screened_stocks