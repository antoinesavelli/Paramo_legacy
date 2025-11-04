# =====================================================
# screener.live.py - Screener for live trading
# =====================================================

from __future__ import annotations
from typing import List, Dict, Union, Optional
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import math

from config import TradingConfig
from data_handler.api import APIDataHandler
from utils import get_logger, log_step

from screener.rules import (
    ScreeningConfigView,
    cfg_view_from,
    Candidate,
    is_price_valid,
    calculate_spread_percent,
    is_float_data_present,
    is_float_shares_valid,
    calculate_relative_volume,
    calculate_momentum_score,
)


class LiveScreener:
    """Live screener implemented using rules helpers (replacing legacy Screener)."""

    def __init__(self, config: TradingConfig, data_handler: Union[APIDataHandler]):
        self.config = config
        self.data_handler = data_handler
        self.logger = get_logger(__name__, component="screener_live")
        self.screened_stocks: List[Dict] = []
        self.last_screen_time: Optional[datetime] = None

    def run_screen(self) -> List[Dict]:
        with log_step(self.logger, "run_screen"):
            universe = self.data_handler.get_universe()
            if universe is None or universe.empty:
                self.logger.info("Universe empty")
                return []

            symbols = universe["symbol"].tolist()
            self.logger.info(f"Universe loaded with {len(symbols)} symbols")

            gaps_df = self.data_handler.calculate_gaps(symbols)
            if gaps_df is None or gaps_df.empty:
                self.logger.info("No gaps computed")
                return []

            min_gap = self.config.screening.MIN_GAP_PERCENT
            gappers = gaps_df[gaps_df["gap_percent"] >= min_gap]
            if gappers.empty:
                self.logger.info("No stocks meeting gap criteria")
                return []

            self.logger.info(f"Gappers meeting criteria: {len(gappers)} (min_gap={min_gap}%)")

            use_threading = len(gappers) > 10
            candidates = self._get_candidates(gappers, use_threading)
            self.logger.info(f"Candidates after checks: {len(candidates)}")

            candidates = self._rank_candidates(candidates)
            candidates = self._apply_sector_limits(candidates)

            self.screened_stocks = [asdict(c) for c in candidates[:10]]
            self.last_screen_time = datetime.now()
            self.logger.info(f"Screening complete. Top candidates: {len(self.screened_stocks)}")
            return self.screened_stocks

    def _get_candidates(self, gappers_df, use_threading: bool) -> List[Candidate]:
        rows = [row for _, row in gappers_df.iterrows()]
        if use_threading:
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(self._analyze_candidate, rows))
        else:
            results = [self._analyze_candidate(r) for r in rows]
        return [r for r in results if r is not None]

    def _analyze_candidate(self, gap_row) -> Optional[Candidate]:
        symbol: str = gap_row["symbol"]
        cfg: ScreeningConfigView = cfg_view_from(self.config)

        try:
            quotes: Dict = self.data_handler.get_quote_data([symbol]) or {}
            quote: Optional[Dict] = quotes.get(symbol)
            if not quote:
                return None

            last_price = quote.get("last")
            if not is_price_valid(last_price, cfg):
                return None

            spread_percent = calculate_spread_percent(quote)
            if not math.isfinite(spread_percent) or spread_percent > cfg.MAX_SPREAD_PERCENT:
                return None

            float_data = self.data_handler.get_float_data(symbol)
            if not is_float_data_present(float_data):
                return None
            if not is_float_shares_valid(float_data.get("float", 0), cfg):
                return None

            daily_bars = self.data_handler.get_intraday_bars(symbol, "1Day", 20)
            rel_vol = calculate_relative_volume(daily_bars, quote.get("volume"))
            if rel_vol is not None and rel_vol < cfg.MIN_RELATIVE_VOLUME:
                return None

            abs_vol = int(quote.get("volume") or 0)
            if abs_vol < cfg.MIN_ABSOLUTE_VOLUME:
                return None

            gap_percent = float(gap_row["gap_percent"])
            momentum_score = calculate_momentum_score(
                gap_percent=gap_percent,
                relative_volume=rel_vol,
                spread_percent=spread_percent,
                absolute_volume=abs_vol,
            )

            return Candidate(
                symbol=symbol,
                gap_percent=gap_percent,
                price=float(last_price),
                volume=abs_vol,
                relative_volume=float(rel_vol or 0.0),
                spread_percent=float(spread_percent),
                momentum_score=float(momentum_score),
                float_shares=float(float_data.get("float")),
                market_cap=float(float_data.get("market_cap") or 0.0),
                timestamp=datetime.now(),
            )
        except (KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def _rank_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        return sorted(candidates, key=lambda x: x.momentum_score, reverse=True)

    def _apply_sector_limits(self, candidates: List[Candidate]) -> List[Candidate]:
        # Not implemented; preserves structure for future diversification rules
        return candidates