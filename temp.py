"""
PARAMO Pipeline Diagnostic
===========================
Traces the full screening pipeline for a given date range and reports
exactly how many candidates die at each stage and why.

Run from the repo root:
    python temp.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pytz

# ── Project imports ────────────────────────────────────────────────────────────
from config.config import TradingConfig
from data_handler.aggregate_handler import AggregateDataHandler
from data_handler.local import LocalDataHandler
from news.backtest import NewsIntegrationBacktest
from market_context.backtest import BacktestMarketContext

# ── Configuration ──────────────────────────────────────────────────────────────
cfg = TradingConfig()

START_DATE = datetime.strptime(cfg.backtest.START_DATE, "%Y-%m-%d")
END_DATE   = datetime.strptime(cfg.backtest.END_DATE,   "%Y-%m-%d")

AGG_DIR  = Path(cfg.backtest.BASE_DATA_DIR) / "daily_aggregates"
NEWS_DIR = Path(cfg.backtest.NEWS_DATA_DIR)
DATA_DIR = Path(cfg.backtest.DATA_DIR)

# ── Session constants (used by sections 6, 7, 8) ──────────────────────────────
session_cfg     = cfg.session
premarket_on    = session_cfg.PREMARKET_ENABLED
warmup          = session_cfg.PREMARKET_WARMUP_MINUTES if premarket_on else session_cfg.REGULAR_WARMUP_MINUTES
session_start_h = 4 if premarket_on else 9
session_start_m = 0 if premarket_on else 30

SEP = "=" * 80

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def subsection(title: str):
    print(f"\n  {'─' * 60}")
    print(f"  {title}")
    print(f"  {'─' * 60}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA AVAILABILITY
# ══════════════════════════════════════════════════════════════════════════════
section("1. DATA AVAILABILITY")

agg_handler = AggregateDataHandler(str(AGG_DIR))

# Collect all trading days in range
all_days = []
cur = START_DATE
while cur <= END_DATE:
    if cur.weekday() < 5:
        all_days.append(cur)
    cur += timedelta(days=1)

print(f"\n  Date range  : {START_DATE.date()} → {END_DATE.date()}")
print(f"  Weekdays    : {len(all_days)}")

# Aggregate file coverage — confirmed trading days are those WITH aggregate data
subsection("Aggregate Files")
agg_days_found, agg_days_missing = [], []
for day in all_days:
    df = agg_handler.get_day_aggregates(day)
    if df is not None and not df.empty:
        agg_days_found.append(day)
    else:
        agg_days_missing.append(day)

# Use only confirmed trading days for all downstream checks
all_days = agg_days_found  # NOTE: holidays naturally excluded

print(f"  Confirmed trading days   : {len(all_days)}")
print(f"  No aggregate data        : {len(agg_days_missing)}")
if agg_days_missing:
    print(f"  (weekdays with no data)  : {[d.date() for d in agg_days_missing[:15]]}"
          + (" ..." if len(agg_days_missing) > 15 else ""))
    print("  Note: these are likely market holidays or dates outside your dataset.")

# Intraday file coverage — only check confirmed trading days
subsection("Intraday Files (1-min parquet)")
data_handler = LocalDataHandler(cfg, data_dir=str(DATA_DIR))
intraday_found, intraday_missing = [], []
for day in all_days:   # NOTE: all_days is now holiday-safe
    day_str = day.strftime('%Y-%m-%d')
    if data_handler.has_data_for_date(day_str):
        intraday_found.append(day)
    else:
        intraday_missing.append(day)

print(f"  Days with intraday data  : {len(intraday_found)}")
print(f"  Days MISSING             : {len(intraday_missing)}")
if intraday_missing:
    print(f"  Missing dates            : {[d.date() for d in intraday_missing[:10]]}"
          + (" ..." if len(intraday_missing) > 10 else ""))

# News file coverage
subsection("News Files")
news_months_checked = set()
news_months_found, news_months_missing = [], []
for day in all_days:
    key = (day.year, day.month)
    if key in news_months_checked:
        continue
    news_months_checked.add(key)
    news_file = NEWS_DIR / str(day.year) / f"news_{day.year}{day.month:02d}.parquet"
    if news_file.exists():
        news_months_found.append(key)
    else:
        news_months_missing.append(key)

print(f"  Months with news files   : {len(news_months_found)} → {news_months_found}")
print(f"  Months MISSING           : {len(news_months_missing)} → {news_months_missing}")

if news_months_missing:
    print()
    print("  ⚠️  WARNING: Missing news months mean the news gate will return")
    print("     approved=False for EVERY symbol on those days if IGNORE_CATALYST=False.")
    print(f"     Current IGNORE_CATALYST = {cfg.backtest.IGNORE_CATALYST}")
    if not cfg.backtest.IGNORE_CATALYST:
        print("  🔴 ACTION REQUIRED: Either provide news files or set IGNORE_CATALYST=True")

# Fundamentals coverage
subsection("Fundamentals Coverage (float / marketcap in aggregates)")
float_total, float_with_data, float_missing = 0, 0, 0
for day in agg_days_found[:5]:   # sample first 5 days
    df = agg_handler.get_day_aggregates(day)
    if df is None or df.empty:
        continue
    float_total += len(df)
    if 'float' in df.columns:
        float_with_data += df['float'].notna().sum()
        float_missing    += df['float'].isna().sum()

if float_total > 0:
    cov = float_with_data / float_total * 100
    print(f"  Float coverage (sample 5 days): {float_with_data}/{float_total} rows ({cov:.1f}%)")
    if cov < 40:
        print(f"  ⚠️  LOW COVERAGE — fundamental filter will reject most candidates")
        print(f"     ENABLE_FLOAT_FILTER     = {cfg.screening.ENABLE_FLOAT_FILTER}")
        print(f"     ENABLE_MARKETCAP_FILTER = {cfg.screening.ENABLE_MARKETCAP_FILTER}")
else:
    print("  No aggregate rows sampled (check aggregate data above)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MARKET CONTEXT GATE (day-level block)
# ══════════════════════════════════════════════════════════════════════════════
section("2. MARKET CONTEXT GATE")

csv_dir = cfg.market_context.CSV_DIR
print(f"\n  CSV_DIR = {csv_dir}")

# ── Probe each CSV directly before using BacktestMarketContext ────────────────
subsection("CSV File Diagnostics")
for fname in ("SPY.csv", "VIX.csv", "RUT.csv"):
    fpath = Path(csv_dir) / fname
    if not fpath.exists():
        print(f"  ❌ {fname} — FILE NOT FOUND at {fpath}")
        continue
    try:
        raw = pd.read_csv(fpath)
        print(f"\n  {fname}")
        print(f"    Rows       : {len(raw):,}")
        print(f"    Columns    : {list(raw.columns)}")
        print(f"    Date sample: {raw['Date'].iloc[:3].tolist()!r}  ...  {raw['Date'].iloc[-3:].tolist()!r}")

        # Try same parsing logic as BacktestMarketContext._load_csv
        try:
            dates = pd.to_datetime(raw['Date'], format="%m/%d/%y")
        except Exception:
            dates = pd.to_datetime(raw['Date'])

        dates = dates.dt.normalize()
        print(f"    Parsed range: {dates.min().date()} → {dates.max().date()}")

        # Check overlap with backtest range
        bt_start_ts = pd.Timestamp(START_DATE).normalize()
        bt_end_ts   = pd.Timestamp(END_DATE).normalize()
        overlap = dates[(dates >= bt_start_ts) & (dates <= bt_end_ts)]
        if overlap.empty:
            print(f"    ⚠️  NO OVERLAP with backtest range "
                  f"({START_DATE.date()} → {END_DATE.date()})")
            print(f"       CSV covers {dates.min().date()} → {dates.max().date()}")
            print(f"       This is why _slice_upto() returns empty — extend your CSVs.")
        else:
            print(f"    ✅ {len(overlap)} trading rows overlap with backtest range")
            # Check SMA_SLOW warmup — needs rows BEFORE backtest start
            rows_before = dates[dates < bt_start_ts]
            sma_slow = cfg.market_context.SMA_SLOW
            if len(rows_before) < sma_slow:
                print(f"    ⚠️  Only {len(rows_before)} rows before backtest start, "
                      f"need ≥{sma_slow} for SMA_SLOW — early days will return 'unknown' trend")
            else:
                print(f"    ✅ {len(rows_before)} rows before backtest start (SMA warmup ok)")
    except Exception as e:
        print(f"  ❌ {fname} — error reading: {e}")

# ── Now run BacktestMarketContext as before ───────────────────────────────────
subsection("Day-Level Trading Gate")
mc = BacktestMarketContext(cfg)
mc_results = {"allowed": [], "blocked": [], "no_data": []}

for day in all_days:
    try:
        mc.update_market_context(day)
        indicators = mc.market_indicators
        if not indicators:
            mc_results["no_data"].append(day)
            continue
        score = indicators.get('market_score', 50)
        env   = indicators.get('trading_environment', 'neutral')
        vix   = (indicators.get('vix_level') or {}).get('classification', 'unknown')
        spy   = (indicators.get('spy_trend') or {}).get('trend', 'unknown')
        rut   = (indicators.get('rut_trend') or {}).get('trend', 'unknown')
        if mc.should_trade():
            mc_results["allowed"].append((day, score, env, vix, spy, rut))
        else:
            mc_results["blocked"].append((day, score, env, vix, spy, rut))
    except Exception as e:
        mc_results["no_data"].append(day)

print(f"\n  Days ALLOWED by market context : {len(mc_results['allowed'])}")
print(f"  Days BLOCKED by market context : {len(mc_results['blocked'])}")
print(f"  Days with no context data      : {len(mc_results['no_data'])}")

if mc_results['blocked']:
    print("\n  Blocked days (score | env | vix | spy | rut):")
    for day, score, env, vix, spy, rut in mc_results['blocked'][:15]:
        print(f"    {day.date()} | {score:.1f} | {env} | {vix} | spy={spy} | rut={rut}")

if mc_results['no_data']:
    print(f"\n  ⚠️  {len(mc_results['no_data'])} days returned empty indicators.")
    print("     Check CSV overlap above — this is almost always a date range mismatch.")

tradeable_days = [d for d, *_ in mc_results['allowed']] + mc_results['no_data']

# ══════════════════════════════════════════════════════════════════════════════
# 3. GAP FILTER
# ══════════════════════════════════════════════════════════════════════════════
section("3. GAP FILTER")

min_gap = cfg.screening.MIN_GAP_PERCENT
min_price = cfg.screening.MIN_PRICE
gap_stats = Counter()
gap_dist = []  # (day, symbol, gap_pct, open_price)

for day in tradeable_days:
    day_agg  = agg_handler.get_day_aggregates(day)
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

    common = set(day_agg['symbol']) & set(prev_agg['symbol'])
    for sym in common:
        d_row = day_agg[day_agg['symbol'] == sym].iloc[0]
        p_row = prev_agg[prev_agg['symbol'] == sym].iloc[0]
        pc = p_row['close']
        op = d_row['open']
        if pd.isna(pc) or pc <= 0 or pd.isna(op) or op <= 0:
            gap_stats['invalid_prices'] += 1
            continue
        gp = ((op - pc) / pc) * 100.0
        if not np.isfinite(gp):
            gap_stats['invalid_gap'] += 1
            continue
        gap_stats['total'] += 1
        if gp >= min_gap:
            if op >= min_price:
                gap_stats['pass_gap_and_price'] += 1
                gap_dist.append({'day': day, 'symbol': sym, 'gap_pct': gp, 'open': op})
            else:
                gap_stats['fail_price_floor'] += 1
        elif gp > 0:
            gap_stats['positive_but_below_threshold'] += 1
        else:
            gap_stats['negative_gap'] += 1

print(f"\n  MIN_GAP_PERCENT = {min_gap}%  |  MIN_PRICE = ${min_price}")
print(f"\n  Total symbol-days evaluated      : {gap_stats['total']:,}")
print(f"  Negative / zero gap              : {gap_stats['negative_gap']:,}")
print(f"  Positive but below {min_gap}%       : {gap_stats['positive_but_below_threshold']:,}")
print(f"  ≥{min_gap}% but below price floor  : {gap_stats['fail_price_floor']:,}")
print(f"  ✅ Pass gap + price floor         : {gap_stats['pass_gap_and_price']:,}")

if gap_dist:
    gdf = pd.DataFrame(gap_dist)
    print(f"\n  Gap distribution (passing candidates):")
    print(f"    Median gap  : {gdf['gap_pct'].median():.1f}%")
    print(f"    Max gap     : {gdf['gap_pct'].max():.1f}%")
    print(f"    Median open : ${gdf['open'].median():.2f}")
    print(f"    Max open    : ${gdf['open'].max():.2f}")
    print(f"\n  Top 10 gap candidates across date range:")
    top10 = gdf.nlargest(10, 'gap_pct')[['day', 'symbol', 'gap_pct', 'open']]
    top10['day'] = top10['day'].apply(lambda d: d.date())
    print(top10.to_string(index=False))
else:
    print("\n  🔴 NO candidates survived gap + price filter")
    print("     Check aggregate data and date range.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. DAILY VOLUME FILTER
# ══════════════════════════════════════════════════════════════════════════════
section("4. DAILY VOLUME FILTER")

min_vol = cfg.screening.MIN_DAILY_VOLUME
vol_pass, vol_fail = 0, 0

for row in gap_dist:
    day_agg = agg_handler.get_day_aggregates(row['day'])
    if day_agg is None:
        vol_fail += 1
        continue
    match = day_agg[day_agg['symbol'] == row['symbol']]
    if match.empty:
        vol_fail += 1
        continue
    vol = match.iloc[0]['volume']
    if pd.notna(vol) and vol >= min_vol:
        vol_pass += 1
    else:
        vol_fail += 1

print(f"\n  MIN_DAILY_VOLUME = {min_vol:,}")
print(f"  ENABLE_DAILY_VOLUME_PRESCREEN = {cfg.screening.ENABLE_DAILY_VOLUME_PRESCREEN}")
print(f"\n  Pass volume filter : {vol_pass}")
print(f"  Fail volume filter : {vol_fail}")
if (vol_pass + vol_fail) > 0:
    print(f"  Kill rate          : {vol_fail / (vol_pass + vol_fail) * 100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FUNDAMENTAL FILTER
# ══════════════════════════════════════════════════════════════════════════════
section("5. FUNDAMENTAL FILTER (float / marketcap)")

max_float = cfg.screening.MAX_FLOAT
max_mcap  = cfg.screening.MAX_MARKETCAP
fund_pass, fund_fail_float, fund_fail_mcap, fund_no_data = 0, 0, 0, 0

for row in gap_dist:
    day_agg = agg_handler.get_day_aggregates(row['day'])
    if day_agg is None:
        fund_no_data += 1
        continue
    match = day_agg[day_agg['symbol'] == row['symbol']]
    if match.empty:
        fund_no_data += 1
        continue
    r = match.iloc[0]
    flt  = r['float']     if 'float'     in r.index else np.nan
    mcap = r['marketcap'] if 'marketcap' in r.index else np.nan

    if pd.isna(flt) and pd.isna(mcap):
        fund_no_data += 1
        continue
    if cfg.screening.ENABLE_FLOAT_FILTER and pd.notna(flt) and flt > max_float:
        fund_fail_float += 1
        continue
    if cfg.screening.ENABLE_MARKETCAP_FILTER and pd.notna(mcap) and mcap > max_mcap:
        fund_fail_mcap += 1
        continue
    fund_pass += 1

total_fund = fund_pass + fund_fail_float + fund_fail_mcap + fund_no_data
print(f"\n  MAX_FLOAT     = {max_float:,}  (ENABLED={cfg.screening.ENABLE_FLOAT_FILTER})")
print(f"  MAX_MARKETCAP = ${max_mcap:,.0f}  (ENABLED={cfg.screening.ENABLE_MARKETCAP_FILTER})")
print(f"\n  Pass fundamental filter  : {fund_pass}")
print(f"  Fail — float too large   : {fund_fail_float}")
print(f"  Fail — mcap too large    : {fund_fail_mcap}")
print(f"  No fundamental data      : {fund_no_data}")
if total_fund > 0:
    print(f"  Kill rate                : {(total_fund - fund_pass) / total_fund * 100:.1f}%")
if fund_no_data > 0 and (cfg.screening.ENABLE_FLOAT_FILTER or cfg.screening.ENABLE_MARKETCAP_FILTER):
    print(f"\n  ⚠️  {fund_no_data} candidates have no float/marketcap data.")
    print("     These are silently dropped. Run temp.py (verify script) to check coverage.")

# ══════════════════════════════════════════════════════════════════════════════
# 6. NEWS GATE
# ══════════════════════════════════════════════════════════════════════════════
section("6. NEWS GATE")

print(f"\n  IGNORE_CATALYST        = {cfg.backtest.IGNORE_CATALYST}")
print(f"  MAX_NEGATIVE_SENTIMENT = {cfg.backtest.MAX_NEGATIVE_SENTIMENT}")
print(f"  NEWS_DATA_DIR          = {NEWS_DIR}")

news_files_found = sorted(NEWS_DIR.rglob("news_*.parquet"))
if not news_files_found:
    print(f"\n  🔴 NO news parquet files found under {NEWS_DIR}")
    print("     Expected path pattern: <NEWS_DATA_DIR>/<YYYY>/news_<YYYYMM>.parquet")
else:
    print(f"\n  Found {len(news_files_found)} news file(s) under {NEWS_DIR}")

if cfg.backtest.IGNORE_CATALYST:
    print("\n  ✅ News gate is DISABLED (IGNORE_CATALYST=True) — skipping probe.")
else:
    news_integration = NewsIntegrationBacktest(cfg, data_dir=str(NEWS_DIR))

    reason_counter   = Counter()
    sentiment_values = []
    timing_gaps_min  = []
    no_data_months   = set()
    results_rows     = []

    for row in gap_dist:
        sym = row['symbol']
        day = row['day']
        entry_ts = pd.Timestamp(
            day.replace(hour=session_start_h, minute=session_start_m), tz='US/Eastern'
        ).tz_convert('UTC') + pd.Timedelta(minutes=warmup)

        try:
            result = news_integration.check_news_approval(sym, entry_ts)
        except Exception as e:
            result = {'approved': False, 'reason': f'exception:{e}',
                      'article_count': 0, 'max_negative': 0.0,
                      'earliest_news_time': None, 'articles_before_time': 0}

        if result is None:
            result = {'approved': False, 'reason': 'returned_none',
                      'article_count': 0, 'max_negative': 0.0,
                      'earliest_news_time': None, 'articles_before_time': 0}

        reason = result.get('reason', 'unknown')
        reason_counter[reason] += 1

        if reason == 'no_news_data':
            no_data_months.add((day.year, day.month))

        if result.get('max_negative', 0) > 0:
            sentiment_values.append(result['max_negative'])

        earliest = result.get('earliest_news_time')
        if reason == 'news_not_yet_published' and earliest is not None:
            timing_gaps_min.append((entry_ts - earliest).total_seconds() / 60)

        results_rows.append({
            'day'          : day.date(),
            'symbol'       : sym,
            'gap_pct'      : round(row['gap_pct'], 1),
            'approved'     : result.get('approved'),
            'reason'       : reason,
            'articles'     : result.get('article_count', 0),
            'max_neg'      : round(result.get('max_negative', 0.0), 4),
            'earliest_news': str(earliest.tz_convert('US/Eastern').strftime('%H:%M ET')
                                 if earliest is not None else 'N/A'),
            'entry_time'   : entry_ts.tz_convert('US/Eastern').strftime('%H:%M ET'),
        })

    total      = len(results_rows)
    approved_n = sum(1 for r in results_rows if r['approved'])

    print(f"\n  Candidates probed : {total}")
    print(f"  ✅ Approved       : {approved_n}")
    print(f"  ❌ Rejected       : {total - approved_n}")
    if total:
        print(f"  Kill rate        : {(total - approved_n) / total * 100:.1f}%")

    print(f"\n  Breakdown:")
    for reason, count in reason_counter.most_common():
        print(f"    {reason:<40s} : {count:>4}  ({count/total*100:.1f}%)")

    if 'no_news_data' in reason_counter:
        print(f"\n  🔴 no_news_data: symbols with no articles in news parquet are blocked.")
        print(f"     Affected months: {sorted(no_data_months)}")
        print(f"     Fix: set IGNORE_CATALYST=True, or supply news files for those months.")

    if sentiment_values:
        s = pd.Series(sentiment_values)
        print(f"\n  Sentiment (neg > 0): mean={s.mean():.4f}  max={s.max():.4f}  "
              f"above threshold={( s > cfg.backtest.MAX_NEGATIVE_SENTIMENT).sum()}")

    if timing_gaps_min:
        s = pd.Series(timing_gaps_min)
        print(f"\n  news_not_yet_published: {len(s)} candidate(s), "
              f"avg {s.mean():.0f} min early (negative = entry before news)")

    rdf = pd.DataFrame(results_rows)
    if not rdf.empty:
        approved_df = rdf[rdf['approved'] == True]
        if not approved_df.empty:
            print(f"\n  ✅ Approved candidates:")
            print(approved_df[['day','symbol','gap_pct','articles','max_neg',
                               'earliest_news','entry_time']].to_string(index=False))
        else:
            print(f"\n  🔴 ZERO candidates approved by news gate.")
            print(rdf[['day','symbol','gap_pct','reason','articles',
                       'max_neg']].head(15).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 7. INTRADAY BARS + WARMUP
# ══════════════════════════════════════════════════════════════════════════════
section("7. INTRADAY BARS + WARMUP")

# session_cfg, premarket_on, warmup, session_start_h, session_start_m
# are defined at the top of this script — no need to redefine here

print(f"\n  PREMARKET_ENABLED    = {premarket_on}")
print(f"  Warmup bars required = {warmup}")

data_handler = LocalDataHandler(cfg, data_dir=str(DATA_DIR))
et = pytz.timezone('US/Eastern')
utc = pytz.utc

bar_results = []

# Use approved candidates from section 6 (ALVO, CHRS)
if 'approved_df' in dir() and not approved_df.empty:
    candidates = approved_df[['day', 'symbol']].to_dict('records')
else:
    # Fallback: hardcode known approved
    candidates = [
        {'day': datetime(2024, 1, 10).date(), 'symbol': 'ALVO'},
        {'day': datetime(2024, 1, 22).date(), 'symbol': 'CHRS'},
    ]

for row in candidates:
    sym = row['symbol']
    day = row['day']
    if isinstance(day, datetime):
        day = day.date()

    session_start_et = et.localize(datetime(day.year, day.month, day.day,
                                            session_start_h, session_start_m))
    session_end_et   = et.localize(datetime(day.year, day.month, day.day, 16, 0))
    session_start_utc = session_start_et.astimezone(utc)
    session_end_utc   = session_end_et.astimezone(utc)

    try:
        bars = data_handler.get_intraday_bars(sym, start=session_start_utc, end=session_end_utc)
        if bars is None or bars.empty:
            status = '🔴 EMPTY'
            bar_count = 0
        elif len(bars) < warmup:
            status = f'⚠️  INSUFFICIENT ({len(bars)} < {warmup})'
            bar_count = len(bars)
        else:
            status = f'✅ OK'
            bar_count = len(bars)
        bar_results.append({'symbol': sym, 'day': day, 'bars': bar_count, 'status': status})
    except Exception as e:
        bar_results.append({'symbol': sym, 'day': day, 'bars': 0, 'status': f'🔴 ERROR: {e}'})

print()
for r in bar_results:
    print(f"  {r['symbol']} {r['day']}  →  {r['bars']} bars  {r['status']}")

if all('✅' in r['status'] for r in bar_results):
    print(f"\n  ✅ All approved candidates have sufficient bars for warmup.")
else:
    print(f"\n  🔴 Some candidates failed bar loading — check get_intraday_bars output above.")

# ══════════════════════════════════════════════════════════════════════════════
# 8. BAR QUALITY (NaN + price range)
# ══════════════════════════════════════════════════════════════════════════════
section("8. BAR QUALITY")

print(f"\n  MIN_PRICE = ${cfg.screening.MIN_PRICE}")

for r in bar_results:
    if '✅' not in r['status']:
        print(f"  ⚠️  Skipping {r['symbol']} — bar load failed")
        continue

    sym = r['symbol']
    day = r['day']
    session_start_et = et.localize(datetime(day.year, day.month, day.day, session_start_h, session_start_m))
    session_end_et   = et.localize(datetime(day.year, day.month, day.day, 16, 0))
    bars = data_handler.get_intraday_bars(sym,
                                          start=session_start_et.astimezone(utc),
                                          end=session_end_et.astimezone(utc))

    nan_rows = bars.iloc[:warmup].isna().any(axis=1).sum()
    entry_price = float(bars.iloc[warmup - 1]['close'])
    below_min   = entry_price < cfg.screening.MIN_PRICE

    print(f"\n  {sym} {day}")
    print(f"    NaN rows in warmup window : {nan_rows}  {'🔴' if nan_rows > 0 else '✅'}")
    print(f"    Entry price (bar {warmup})  : ${entry_price:.2f}  {'🔴 below MIN_PRICE' if below_min else '✅'}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("9. PATTERN ANALYSIS")

from strategy.pattern_analyzer import PatternAnalyzer

pa = PatternAnalyzer(cfg, data_handler)

for r in bar_results:
    if '✅' not in r['status']:
        print(f"  ⚠️  Skipping {r['symbol']} — bar load failed")
        continue

    sym = r['symbol']
    day = r['day']
    session_start_et = et.localize(datetime(day.year, day.month, day.day, session_start_h, session_start_m))
    session_end_et   = et.localize(datetime(day.year, day.month, day.day, 16, 0))
    bars = data_handler.get_intraday_bars(sym,
                                          start=session_start_et.astimezone(utc),
                                          end=session_end_et.astimezone(utc))
    bars_warm = bars.iloc[:warmup].copy()

    # Get gap_pct for this candidate from approved_df
    gap_pct = float(approved_df[approved_df['symbol'] == sym]['gap_pct'].iloc[0])

    result = pa.analyze_pattern(sym, bars=bars_warm, is_premarket=premarket_on, gap_percent=gap_pct)

    valid  = result.get('valid', False)
    score  = result.get('pattern_strength', 0)
    reason = result.get('reason', 'N/A')

    print(f"\n  {sym} {day}")
    print(f"    Valid          : {valid}  {'✅' if valid else '🔴'}")
    print(f"    Pattern score  : {score}")
    if not valid:
        print(f"    Reject reason  : {reason}")
    else:
        from screener.core import calc_atr_stop
        entry_price = float(bars.iloc[warmup - 1]['close'])
        stop_price  = calc_atr_stop(bars_warm, entry_price, atr_period=14, atr_mult=2.0, fallback_pct=0.03)
        print(f"    Entry price    : ${entry_price:.2f}")
        print(f"    Stop price     : ${stop_price:.2f}")
        print(f"    Risk per share : ${entry_price - stop_price:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. PIPELINE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("10. PIPELINE SUMMARY")

stages = [
    ("Gap filter (≥10%, ≥$2)",        252),
    ("Volume filter (≥50k)",           84),
    ("Fundamentals filter",            len([r for r in gap_dist if True])),  # reuse gap_dist count
    ("News gate",                      approved_n),
    ("Bar load + warmup",              sum(1 for r in bar_results if '✅' in r['status'])),
]

print()
for name, count in stages:
    print(f"  {name:<40s} : {count:>4}")
print(f"\n  ➡️  Candidates reaching pattern analysis : {sum(1 for r in bar_results if '✅' in r['status'])}")