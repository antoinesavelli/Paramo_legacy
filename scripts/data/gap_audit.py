# =====================================================
# scripts/gap_audit.py - Gap Band Audit
# =====================================================
"""
Counts how many qualifying gappers fell into each gap-% band over the
backtest date range, applying every ScreeningConfig filter:
  - MIN_GAP_PERCENT
  - MIN_PRICE
  - MIN_FLOAT / MAX_FLOAT  (if ENABLE_FLOAT_FILTER)
  - MAX_MARKETCAP          (if ENABLE_MARKETCAP_FILTER)
  - MIN_DAILY_VOLUME       (if ENABLE_DAILY_VOLUME_PRESCREEN)

Gap bands are tuned for small-cap momentum gappers:
  10–20 %  |  20–30 %  |  30–50 %  |  50–75 %  |  75–100 %  |  100–150 %  |  150–200 %  |  200 %+

Usage:
    python -m scripts.gap_audit
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict

from config.config import (
    BacktestConfig,
    ScreeningConfig,
)
from data_handler.aggregates.aggregate_handler import AggregateDataHandler

# ── band definitions (right-exclusive except the last) ───────────────────────
BANDS: list[tuple[float, float, str]] = [
    (10,   20,  " 10–20 %"),
    (20,   30,  " 20–30 %"),
    (30,   50,  " 30–50 %"),
    (50,   75,  " 50–75 %"),
    (75,  100,  " 75–100%"),
    (100, 150,  "100–150%"),
    (150, 200,  "150–200%"),
    (200, float("inf"), "  200 %+"),
]


def _assign_band(gap_pct: float) -> str | None:
    """Return the band label for a gap %, or None if below first band edge."""
    for lo, hi, label in BANDS:
        if lo <= gap_pct < hi:
            return label
    return None


def _trading_days(start: str, end: str, agg: AggregateDataHandler) -> list[pd.Timestamp]:
    """Return every calendar date in [start, end] that has aggregate data."""
    days = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        if cur.weekday() < 5:                       # Mon–Fri only
            days.append(cur)
        cur += timedelta(days=1)
    return days


def run_audit() -> None:
    bt   = BacktestConfig()
    scr  = ScreeningConfig()
    agg  = AggregateDataHandler(
        str(Path(bt.BASE_DATA_DIR) / "daily_aggregates"),
        cache_limit=3,
    )

    # ── per-band counters ─────────────────────────────────────────────────────
    band_counts:      dict[str, int] = defaultdict(int)
    band_days_seen:   dict[str, set] = defaultdict(set)   # unique trading days per band
    total_days        = 0
    days_with_gappers = 0

    calendar = _trading_days(bt.START_DATE, bt.END_DATE, agg)

    for cur in calendar:
        cur_df = agg.get_day_aggregates(cur)
        if cur_df is None or cur_df.empty:
            continue

        # find previous trading day (up to 7 calendar days back)
        prev_df = None
        for offset in range(1, 8):
            prev_df = agg.get_day_aggregates(cur - timedelta(days=offset))
            if prev_df is not None and not prev_df.empty:
                break
        if prev_df is None or prev_df.empty:
            continue

        total_days += 1

        # ── merge on common symbols ───────────────────────────────────────────
        prev_close = prev_df[["symbol", "close"]].rename(columns={"close": "prev_close"})
        merged = cur_df.merge(prev_close, on="symbol", how="inner")

        # ── gap % ─────────────────────────────────────────────────────────────
        valid_prev = merged["prev_close"] > 0
        valid_open = merged["open"] > 0
        merged = merged[valid_prev & valid_open].copy()
        merged["gap_pct"] = (
            (merged["open"] - merged["prev_close"]) / merged["prev_close"]
        ) * 100.0
        merged = merged[np.isfinite(merged["gap_pct"])]

        # ── apply config screening filters ───────────────────────────────────
        mask = (
            (merged["gap_pct"]  >= scr.MIN_GAP_PERCENT) &
            (merged["open"]     >= scr.MIN_PRICE)
        )

        if scr.ENABLE_DAILY_VOLUME_PRESCREEN and "volume" in merged.columns:
            mask &= merged["volume"] >= scr.MIN_DAILY_VOLUME

        if scr.ENABLE_FLOAT_FILTER and "float" in merged.columns:
            float_col = pd.to_numeric(merged["float"], errors="coerce")
            if scr.MIN_FLOAT > 0:
                mask &= float_col.fillna(0) >= scr.MIN_FLOAT
            mask &= float_col.fillna(float("inf")) <= scr.MAX_FLOAT

        if scr.ENABLE_MARKETCAP_FILTER and "marketcap" in merged.columns:
            mc_col = pd.to_numeric(merged["marketcap"], errors="coerce")
            mask &= mc_col.fillna(float("inf")) <= scr.MAX_MARKETCAP

        qualified = merged[mask]

        if qualified.empty:
            continue

        days_with_gappers += 1
        day_str = str(cur.date())

        for gap_pct in qualified["gap_pct"]:
            label = _assign_band(gap_pct)
            if label:
                band_counts[label] += 1
                band_days_seen[label].add(day_str)

    # ── print results ─────────────────────────────────────────────────────────
    total_gappers = sum(band_counts.values())

    print("\n" + "=" * 60)
    print(" GAP AUDIT RESULTS")
    print(f" Period  : {bt.START_DATE}  →  {bt.END_DATE}")
    print(f" Filters : MIN_GAP={scr.MIN_GAP_PERCENT}%  MIN_PRICE=${scr.MIN_PRICE}"
          f"  MAX_FLOAT={scr.MAX_FLOAT:,}  MAX_MCAP=${scr.MAX_MARKETCAP/1e6:.0f}M"
          f"  MIN_VOL={scr.MIN_DAILY_VOLUME:,}")
    print("=" * 60)
    print(f"  Trading days in range  : {total_days:>6,}")
    print(f"  Days with ≥1 gapper    : {days_with_gappers:>6,}")
    print(f"  Total qualifying gaps  : {total_gappers:>6,}")
    print("-" * 60)
    print(f"  {'Band':<12}  {'Count':>7}  {'% of total':>10}  {'Avg/day':>8}  {'Days active':>12}")
    print("-" * 60)

    for lo, hi, label in BANDS:
        count     = band_counts.get(label, 0)
        pct_total = (count / total_gappers * 100) if total_gappers else 0.0
        days_act  = len(band_days_seen.get(label, set()))
        avg_day   = (count / days_with_gappers) if days_with_gappers else 0.0
        print(f"  {label:<12}  {count:>7,}  {pct_total:>9.1f}%  {avg_day:>8.2f}  {days_act:>12,}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_audit()