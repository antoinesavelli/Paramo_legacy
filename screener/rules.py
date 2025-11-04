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
    MAX_SPREAD_PERCENT: float
    MIN_FLOAT: int
    MAX_FLOAT: int


def cfg_view_from(config) -> ScreeningConfigView:
    s = config.screening
    return ScreeningConfigView(
        MIN_GAP_PERCENT=s.MIN_GAP_PERCENT,
        MIN_PRICE=s.MIN_PRICE,
        MAX_PRICE=s.MAX_PRICE,
        MIN_RELATIVE_VOLUME=s.MIN_RELATIVE_VOLUME,
        MIN_ABSOLUTE_VOLUME=s.MIN_ABSOLUTE_VOLUME,
        MAX_SPREAD_PERCENT=s.MAX_SPREAD_PERCENT,
        MIN_FLOAT=s.MIN_FLOAT,
        MAX_FLOAT=s.MAX_FLOAT
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
    spread_percent: float
    momentum_score: float
    float_shares: float
    market_cap: float
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


def calculate_spread_percent(quote: Dict) -> float:
    """
    Computes spread% = (ask - bid) / last * 100.
    If data is missing or invalid, returns +inf to force rejection upstream.
    """
    try:
        last = float(quote.get("last", 0) or 0)
        bid = float(quote.get("bid", 0) or 0)
        ask = float(quote.get("ask", 0) or 0)
        if last <= 0 or bid <= 0 or ask <= 0 or ask < bid:
            return float("inf")
        return (ask - bid) / last * 100.0
    except Exception:
        return float("inf")


def is_float_data_present(float_data: Optional[Dict]) -> bool:
    """
    Checks if float data structure is present and contains a numeric 'float'.
    """
    if not float_data:
        return False
    try:
        f = float(float_data.get("float", 0))
        return f > 0
    except Exception:
        return False


def is_float_shares_valid(float_shares: float, cfg: ScreeningConfigView) -> bool:
    try:
        fs = float(float_shares)
        return cfg.MIN_FLOAT <= fs <= cfg.MAX_FLOAT
    except Exception:
        return False


def calculate_relative_volume(daily_bars: pd.DataFrame, current_volume: Optional[float]) -> Optional[float]:
    """
    Computes relative volume as current total volume vs average daily volume over recent periods.
    Expects daily bars with a 'volume' column. If the last row is today's bar, we exclude it
    from the average for a more conservative baseline.
    """
    if daily_bars is None or daily_bars.empty or "volume" not in daily_bars.columns:
        return None
    try:
        vols = daily_bars["volume"].astype(float)
        baseline = vols.iloc[:-1].mean() if len(vols) >= 2 else vols.mean()
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
    spread_percent: float,
    absolute_volume: Optional[float]
) -> float:
    """
    Heuristic momentum score combining gap%, relative volume, spread penalty, and size.
    Tuned to be stable without external deps.
    """
    rv = max(0.0, min(float(relative_volume or 0.0), 10.0))
    sp = max(0.0, float(spread_percent if math.isfinite(spread_percent) else 100.0))
    av = max(1.0, float(absolute_volume or 1.0))
    size_term = min(math.log10(av), 7.0) * 0.3
    return float(gap_percent) * 1.0 + rv * 2.0 - sp * 0.5 + size_term
