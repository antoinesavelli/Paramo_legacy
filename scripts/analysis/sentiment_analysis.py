"""
Sentiment Distribution Analysis
================================
Computes statistical distribution of neg / pos / polarity scores to help
determine optimal gate thresholds.

TOGGLES (edit the CONFIG block below):
    FILTER_MODE   : 'all'     → all articles
                    'gappers' → gapper symbols only
    DATE_RANGE    : 'backtest' → use START_DATE / END_DATE from BacktestConfig
                    'full'     → entire news dataset (all files, no date filter)

Run from repo root:
    python scripts/sentiment_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from config.config import TradingConfig
from data_handler.aggregates.aggregate_handler import AggregateDataHandler

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these two toggles
# ══════════════════════════════════════════════════════════════════════════════
FILTER_MODE = 'all'   # 'all' | 'gappers'
DATE_RANGE  = 'full'  # 'backtest' | 'full'
# ══════════════════════════════════════════════════════════════════════════════

cfg      = TradingConfig()
NEWS_DIR = Path(cfg.backtest.NEWS_DATA_DIR)
AGG_DIR  = Path(cfg.backtest.BASE_DATA_DIR) / "daily_aggregates"
SEP      = "=" * 80

START_DATE   = datetime.strptime(cfg.backtest.START_DATE, "%Y-%m-%d")
END_DATE     = datetime.strptime(cfg.backtest.END_DATE,   "%Y-%m-%d")
bt_start_str = START_DATE.strftime('%Y-%m-%d')
bt_end_str   = END_DATE.strftime('%Y-%m-%d')

SENTIMENT_COLS = {
    'neg':      ['neg', 'negative'],
    'pos':      ['pos', 'positive'],
    'polarity': ['polarity', 'compound', 'sentiment_score'],
}

PERCENTILES = [25, 50, 75, 80, 85, 90, 95, 99]

# ── Build gapper symbol set (only needed for 'gappers' mode) ──────────────────
gap_syms = set()
if FILTER_MODE == 'gappers':
    import calendar
    print("Building gap candidates...")
    agg_handler = AggregateDataHandler(str(AGG_DIR))

    if DATE_RANGE == 'full':
        agg_files = sorted(AGG_DIR.glob("**/????-??.parquet"))
        if agg_files:
            def _parse_agg(p: Path) -> datetime:
                return datetime.strptime(p.stem, "%Y-%m")
            scan_start = _parse_agg(agg_files[0]).replace(day=1)
            _last      = _parse_agg(agg_files[-1])
            scan_end   = _last.replace(day=calendar.monthrange(_last.year, _last.month)[1])
            print(f"  Full dataset range: {scan_start.date()} → {scan_end.date()}")
        else:
            scan_start, scan_end = START_DATE, END_DATE
    else:
        scan_start, scan_end = START_DATE, END_DATE

    all_days = []
    cur = scan_start
    while cur <= scan_end:
        if cur.weekday() < 5:
            all_days.append(cur)
        cur += timedelta(days=1)

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

        day_map  = {str(r['symbol']): float(r['open']  or 0) for r in day_agg.drop_duplicates('symbol').to_dict('records')}
        prev_map = {str(r['symbol']): float(r['close'] or 0) for r in prev_agg.drop_duplicates('symbol').to_dict('records')}

        for sym in set(day_map) & set(prev_map):
            op = day_map[sym]
            cl = prev_map[sym]
            if not op or not cl or cl <= 0 or op <= 0:
                continue
            gp = ((op - cl) / cl) * 100.0
            if gp >= cfg.screening.MIN_GAP_PERCENT and op >= cfg.screening.MIN_PRICE:
                gap_syms.add(sym)

    print(f"Gap candidates: {len(gap_syms)} unique symbols\n")

gap_set_arrow = pa.array(sorted(gap_syms), type=pa.string()) if gap_syms else None

# ── Collect ────────────────────────────────────────────────────────────────────
news_files = sorted(NEWS_DIR.glob("**/news_*.parquet"))
print(f"Found {len(news_files)} news files. Scanning...")

buffers    = {k: [] for k in SENTIMENT_COLS}
skipped    = 0
total_rows = 0

for news_file in news_files:
    try:
        schema  = pq.read_schema(news_file)
        columns = schema.names
    except Exception:
        skipped += 1
        continue

    date_col = next((c for c in ['date', 'published_at'] if c in columns), None)
    sym_col  = next((c for c in ['ticker', 'symbol']     if c in columns), None)
    file_cols = {}
    for metric, candidates in SENTIMENT_COLS.items():
        found = next((c for c in candidates if c in columns), None)
        if found:
            file_cols[metric] = found

    if not file_cols:
        skipped += 1
        continue

    read_cols = list(dict.fromkeys(
        ([date_col] if date_col and DATE_RANGE == 'backtest' else []) +
        ([sym_col]  if sym_col  and FILTER_MODE == 'gappers' else []) +
        list(file_cols.values())
    ))

    try:
        table = pq.read_table(news_file, columns=read_cols)
    except Exception:
        skipped += 1
        continue

    # ── Date range filter ──────────────────────────────────────────────────
    if DATE_RANGE == 'backtest' and date_col:
        date_arr = table.column(date_col)
        if pa.types.is_timestamp(date_arr.type):
            date_only = pc.cast(date_arr, pa.date32())
        else:
            date_only = pc.cast(pc.cast(date_arr, pa.string()), pa.date32())
        lo = pc.cast(pc.strptime(bt_start_str, format='%Y-%m-%d', unit='s'), pa.date32())
        hi = pc.cast(pc.strptime(bt_end_str,   format='%Y-%m-%d', unit='s'), pa.date32())
        date_mask = pc.and_(pc.greater_equal(date_only, lo), pc.less_equal(date_only, hi))
        table = table.filter(date_mask)

    # ── Symbol filter ──────────────────────────────────────────────────────
    if FILTER_MODE == 'gappers' and sym_col and gap_set_arrow is not None:
        gap_mask = pc.is_in(table.column(sym_col), value_set=gap_set_arrow)
        table = table.filter(gap_mask)

    if table.num_rows == 0:
        continue

    total_rows += table.num_rows

    for metric, col in file_cols.items():
        if col not in table.schema.names:
            continue
        arr = pc.cast(pc.drop_null(table.column(col)), pa.float64())
        buffers[metric].append(arr)

print(f"Loaded {total_rows:,} rows from {len(news_files) - skipped} files "
      f"({skipped} skipped).\n")

# ── Stats ──────────────────────────────────────────────────────────────────────
def stats_block(data: np.ndarray) -> None:
    pct_vals     = np.percentile(data, PERCENTILES)
    above_counts = {p: int(np.sum(data > v)) for p, v in zip(PERCENTILES, pct_vals)}

    print(f"    Samples   : {data.size:,}")
    print(f"    Mean      : {data.mean():.6f}")
    print(f"    Median    : {np.median(data):.6f}")
    print(f"    Std Dev   : {data.std():.6f}")
    print(f"    Min / Max : {data.min():.6f} / {data.max():.6f}")
    z = (data - data.mean()) / (data.std() + 1e-12)
    print(f"    Skewness  : {float(np.mean(z ** 3)):.4f}")
    print(f"    Kurtosis  : {float(np.mean(z ** 4) - 3):.4f}")
    print()
    print(f"    {'Percentile':<12} {'Value':>10}   {'Above':>8}   {'% pass':>7}  Chart")
    print(f"    {'-'*58}")
    for p, v, above in zip(PERCENTILES, pct_vals, above_counts.values()):
        bar = "█" * int((above / data.size) * 20)
        print(f"    p{p:<11} {v:>10.6f}   {above:>8,}   {above/data.size*100:6.1f}%  {bar}")

    suggested = next(
        ((p, v) for p, v in zip(PERCENTILES, pct_vals) if above_counts[p] / data.size < 0.15),
        (PERCENTILES[-1], pct_vals[-1])
    )
    print(f"\n    ► Suggested gate (< 15% pass): p{suggested[0]} → {suggested[1]:.4f}")

# ── Report ─────────────────────────────────────────────────────────────────────
filter_label = (f"GAPPERS ONLY  ({len(gap_syms)} syms, backtest range)"
                if FILTER_MODE == 'gappers'
                else "ALL ARTICLES")
date_label   = f"{bt_start_str} → {bt_end_str}" if DATE_RANGE == 'backtest' else "FULL DATASET"

print(f"\n{SEP}\n  SENTIMENT DISTRIBUTION ANALYSIS\n{SEP}")
print(f"  News directory : {NEWS_DIR}")
print(f"  Filter mode    : {filter_label}")
print(f"  Date range     : {date_label}")
print(f"  Files scanned  : {len(news_files) - skipped} / {len(news_files)}")
print(f"  Rows analysed  : {total_rows:,}")

for metric, chunks in buffers.items():
    print(f"\n{SEP}")
    print(f"  METRIC: {metric.upper()}")
    if not chunks:
        print("    no data found")
        continue
    combined = np.concatenate([c.to_pylist() for c in chunks]).astype(np.float64)
    print()
    stats_block(combined)

print(f"\n{SEP}")
print("  Set MAX_NEGATIVE_SENTIMENT in BacktestConfig to the neg gate value above.")
print(SEP)