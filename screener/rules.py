# =====================================================
# screener.rules.py - Screener Rule Set
# =====================================================

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import math
import pandas as pd


# -------------------------------
# Config view
# -------------------------------
@dataclass(frozen=True)
class ScreeningConfigView:
    MIN_GAP_PERCENT: float
    MIN_PRICE: float
    MAX_PRICE: float
    MIN_RELATIVE_VOLUME: float
    MIN_ABSOLUTE_VOLUME: int


def cfg_view_from(config) -> ScreeningConfigView:
    s = config.screening
    return ScreeningConfigView(
        MIN_GAP_PERCENT=s.MIN_GAP_PERCENT,
        MIN_PRICE=s.MIN_PRICE,
        MAX_PRICE=s.MAX_PRICE,
        MIN_RELATIVE_VOLUME=s.MIN_RELATIVE_VOLUME,
        MIN_ABSOLUTE_VOLUME=s.MIN_ABSOLUTE_VOLUME
    )


def filter_price_and_gap(gaps_df: pd.DataFrame, cfg: ScreeningConfigView) -> pd.DataFrame:
    if gaps_df is None or gaps_df.empty:
        return pd.DataFrame(columns=["symbol", "gap_percent", "last_price", "open_price", "prev_close"])
    m = (
        (gaps_df["gap_percent"] >= cfg.MIN_GAP_PERCENT) &
        (gaps_df["last_price"] >= cfg.MIN_PRICE) &
        (gaps_df["last_price"] <= cfg.MAX_PRICE)
    )
    return gaps_df.loc[m].copy()


# -------------------------------
# Shared DTO
# -------------------------------
@dataclass
class Candidate:
    symbol: str
    gap_percent: float
    price: float
    volume: int
    relative_volume: float
    momentum_score: float
    timestamp: datetime


# -------------------------------
# Core rule helpers (pure)
# -------------------------------
def is_price_valid(last_price: Optional[float], cfg: ScreeningConfigView) -> bool:
    try:
        if last_price is None:
            return False
        p = float(last_price)
        return cfg.MIN_PRICE <= p <= cfg.MAX_PRICE
    except Exception:
        return False


def calculate_relative_volume(daily_bars: pd.DataFrame, current_volume: Optional[float]) -> Optional[float]:
    """Computes relative volume, excluding zero-volume days from baseline."""
    if daily_bars is None or daily_bars.empty or "volume" not in daily_bars.columns:
        return None
    try:
        vols = daily_bars["volume"].astype(float)
        
        # ✅ Exclude zero-volume days (halts, no trading)
        valid_vols = vols[vols > 0]
        
        if len(valid_vols) < 5:  # Need at least 5 valid days
            return None
        
        baseline = valid_vols.mean()
        
        if not baseline or baseline <= 0:
            return None
        
        cv = float(current_volume or 0)
        if cv <= 0:
            return None
        
        return cv / baseline
    except Exception:
        return None


def calculate_momentum_score(
    gap_percent: float,
    relative_volume: Optional[float],
    absolute_volume: Optional[float]
) -> float:
    """
    Heuristic momentum score combining gap%, relative volume, and size.
    Tuned to be stable without external deps.
    """
    rv = max(0.0, min(float(relative_volume or 0.0), 10.0))
    av = max(1.0, float(absolute_volume or 1.0))
    size_term = min(math.log10(av), 7.0) * 0.3
    return float(gap_percent) * 1.0 + rv * 2.0 + size_term
