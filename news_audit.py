"""
News File Audit
===============
Examines news parquet files and reports coverage against gap candidates.

Run from repo root:
    python news_audit.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from config.config import TradingConfig
from data_handler.aggregate_handler import AggregateDataHandler

cfg        = TradingConfig()
START_DATE = datetime.strptime(cfg.backtest.START_DATE, "%Y-%m-%d")
END_DATE   = datetime.strptime(cfg.backtest.END_DATE,   "%Y-%m-%d")
AGG_DIR    = Path(cfg.backtest.BASE_DATA_DIR) / "daily_aggregates"
NEWS_DIR   = Path(cfg.backtest.NEWS_DATA_DIR)

SEP = "=" * 80

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ── Build gap candidates (same logic as temp.py section 3) ────────────────────
agg_handler = AggregateDataHandler(str(AGG_DIR))

all_days = []
cur = START_DATE
while cur <= END_DATE:
    if cur.weekday() < 5:
        all_days.append(cur)
    cur += timedelta(days=1)

gap_dist = []
for day in all_days:
    day_agg = agg_handler.get_day_aggregates(day)
    if day_agg is None or day_agg.empty:
        continue
    prev_day = day - timedelta(days=1)
    for _ in range(7):
        prev_agg = agg_handler.get_day_aggregates(prev_day)
        if prev_agg is not None and not prev_agg.empty:
            break
        prev_day -= timedelta(days=1)
    else:
        continue
    for sym in set(day_agg['symbol']) & set(prev_agg['symbol']):
        d_row = day_agg[day_agg['symbol'] == sym].iloc[0]
        p_row = prev_agg[prev_agg['symbol'] == sym].iloc[0]
        pc, op = p_row['close'], d_row['open']
        if pd.isna(pc) or pc <= 0 or pd.isna(op) or op <= 0:
            continue
        gp = ((op - pc) / pc) * 100.0
        if gp >= cfg.screening.MIN_GAP_PERCENT and op >= cfg.screening.MIN_PRICE:
            gap_dist.append({'symbol': sym, 'day': day, 'gap_pct': gp})

gap_syms = set(r['symbol'] for r in gap_dist)

# ── Aggregate across all files ────────────────────────────────────────────────
all_matched_syms = set()
total_rows       = 0
total_in_range   = 0
files_with_range = 0
files_missing_cols = []
col_issues       = []
sentiment_vals   = []

bt_start_ts = pd.Timestamp(START_DATE, tz='US/Eastern').normalize()
bt_end_ts   = pd.Timestamp(END_DATE,   tz='US/Eastern').normalize()

for news_file in news_files:
    try:
        df = pd.read_parquet(news_file)
    except Exception as e:
        col_issues.append(f"load error: {news_file.name}")
        continue

    date_col = next((c for c in ['date', 'published_at'] if c in df.columns), None)
    sym_col  = next((c for c in ['ticker', 'symbol']     if c in df.columns), None)
    neg_col  = next((c for c in ['negative', 'neg']      if c in df.columns), None)

    if not date_col or not sym_col:
        col_issues.append(news_file.name)
        continue

    total_rows += len(df)

    try:
        dates = pd.to_datetime(df[date_col], utc=True).dt.tz_convert('US/Eastern').dt.normalize()
        in_range = (dates >= bt_start_ts) & (dates <= bt_end_ts)
        total_in_range += in_range.sum()
        if in_range.sum() > 0:
            files_with_range += 1
            overlap = gap_syms & set(df.loc[in_range, sym_col].unique())
            all_matched_syms |= overlap
    except Exception:
        pass

    if neg_col:
        sentiment_vals.extend(df[neg_col].dropna().tolist())

section("NEWS FILE AUDIT")

print(f"\n  Backtest range           : {START_DATE.date()} → {END_DATE.date()}")
print(f"  News files found         : {len(news_files)}")
print(f"  Files with backtest data : {files_with_range} / {len(news_files)}")
print(f"  Total articles           : {total_rows:,}")
print(f"  Articles in backtest range : {total_in_range:,}")

if col_issues:
    print(f"  ⚠️  Files with missing columns : {len(col_issues)}")

if sentiment_vals:
    s = pd.Series(sentiment_vals)
    above = (s > cfg.backtest.MAX_NEGATIVE_SENTIMENT).sum()
    print(f"\n  Sentiment  : mean={s.mean():.4f}  max={s.max():.4f}  above threshold={above} ({above/len(s)*100:.1f}%)")

if gap_syms:
    covered   = len(all_matched_syms)
    uncovered = len(gap_syms) - covered
    print(f"\n  Gap candidates (unique)  : {len(gap_syms)}")
    print(f"  Covered by news          : {covered} ({covered/len(gap_syms)*100:.1f}%)")
    print(f"  Not covered              : {uncovered} ({uncovered/len(gap_syms)*100:.1f}%)")
    if all_matched_syms:
        print(f"  Example covered          : {sorted(all_matched_syms)[:2]}")
else:
    print("\n  🔴 No gap candidates built — check aggregate data.")