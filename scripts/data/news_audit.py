"""
News File Audit
===============
Examines news parquet files and reports coverage against gap candidates.

Run from repo root:
    python scripts/news_audit.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pa_compute
from pathlib import Path
from datetime import datetime, timedelta

from config.config import TradingConfig
from data_handler.aggregate_handler import AggregateDataHandler

cfg        = TradingConfig()
START_DATE = datetime.strptime(cfg.backtest.START_DATE, "%Y-%m-%d")
END_DATE   = datetime.strptime(cfg.backtest.END_DATE,   "%Y-%m-%d")
AGG_DIR    = Path(cfg.backtest.BASE_DATA_DIR) / "daily_aggregates"
NEWS_DIR   = Path(cfg.backtest.NEWS_DATA_DIR)

bt_start_str = START_DATE.strftime('%Y-%m-%d')
bt_end_str   = END_DATE.strftime('%Y-%m-%d')

SEP = "=" * 80

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ── Build gap candidates ───────────────────────────────────────────────────────
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

    # Convert to plain dicts — avoids all pandas Arrow/.at[] hanging issues
    day_map  = {str(r['symbol']): float(r['open']  or 0) for r in day_agg.drop_duplicates('symbol').to_dict('records')}
    prev_map = {str(r['symbol']): float(r['close'] or 0) for r in prev_agg.drop_duplicates('symbol').to_dict('records')}

    for sym in set(day_map) & set(prev_map):
        op = day_map[sym]
        pc = prev_map[sym]
        if not op or not pc or pc <= 0 or op <= 0:
            continue
        gp = ((op - pc) / pc) * 100.0
        if gp >= cfg.screening.MIN_GAP_PERCENT and op >= cfg.screening.MIN_PRICE:
            gap_dist.append({'symbol': sym, 'day': day, 'gap_pct': gp})

gap_syms = set(r['symbol'] for r in gap_dist)
print(f"Gap candidates: {len(gap_syms)} unique symbols across {len(gap_dist)} day/symbol pairs")

# ── Scan news files via pyarrow only ──────────────────────────────────────────
all_matched_syms = set()
total_rows       = 0
total_in_range   = 0
files_with_range = 0
col_issues       = []
sentiment_vals   = []

news_files = sorted(NEWS_DIR.glob("**/news_*.parquet"))
print(f"Scanning {len(news_files)} news files...")

for news_file in news_files:
    try:
        schema  = pq.read_schema(news_file)
        columns = schema.names
    except Exception as e:
        col_issues.append(f"schema error: {news_file.name}: {e}")
        continue

    date_col = next((c for c in ['date', 'published_at'] if c in columns), None)
    sym_col  = next((c for c in ['ticker', 'symbol']     if c in columns), None)
    neg_col  = next((c for c in ['negative', 'neg']      if c in columns), None)

    if not date_col or not sym_col:
        col_issues.append(f"missing cols: {news_file.name} — found: {columns[:8]}")
        continue

    try:
        read_cols = [date_col, sym_col] + ([neg_col] if neg_col else [])
        table     = pq.read_table(news_file, columns=read_cols)
    except Exception as e:
        col_issues.append(f"read error: {news_file.name}: {e}")
        continue

    total_rows += table.num_rows

    date_col_arr = table.column(date_col)
    # Normalise to date32 for fast range filtering
    if pa.types.is_timestamp(date_col_arr.type):
        date_only = pa_compute.cast(date_col_arr, pa.date32())
    else:
        date_only = pa_compute.cast(pa_compute.cast(date_col_arr, 'string'), pa.date32())

    lo = pa_compute.strptime(bt_start_str, format='%Y-%m-%d', unit='s')
    hi = pa_compute.strptime(bt_end_str,   format='%Y-%m-%d', unit='s')
    lo_date = pa_compute.cast(lo, pa.date32())
    hi_date = pa_compute.cast(hi, pa.date32())

    mask = pa_compute.and_(
        pa_compute.greater_equal(date_only, lo_date),
        pa_compute.less_equal(date_only,    hi_date),
    )
    filtered       = table.filter(mask)
    count_in_range = filtered.num_rows
    in_range_syms  = set(v for v in filtered.column(sym_col).to_pylist() if v)

    total_in_range += count_in_range
    if count_in_range > 0:
        files_with_range += 1
        all_matched_syms |= (gap_syms & in_range_syms)

    if neg_col:
        sentiment_vals.extend(v for v in table.column(neg_col).to_pylist() if v is not None)

# ── Report ─────────────────────────────────────────────────────────────────────
section("NEWS FILE AUDIT")

print(f"\n  Backtest range             : {START_DATE.date()} → {END_DATE.date()}")
print(f"  News files found           : {len(news_files)}")
print(f"  Files with backtest data   : {files_with_range} / {len(news_files)}")
print(f"  Total articles             : {total_rows:,}")
print(f"  Articles in backtest range : {total_in_range:,}")

if col_issues:
    print(f"  ⚠️  Files with issues        : {len(col_issues)}")
    for issue in col_issues[:2]:
        print(f"      {issue}")

if sentiment_vals:
    n     = len(sentiment_vals)
    mean  = sum(sentiment_vals) / n
    maxv  = max(sentiment_vals)
    above = sum(1 for v in sentiment_vals if v > cfg.backtest.MAX_NEGATIVE_SENTIMENT)
    print(f"\n  Sentiment : mean={mean:.4f}  max={maxv:.4f}  above threshold={above} ({above/n*100:.1f}%)")

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