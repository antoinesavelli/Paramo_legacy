# =====================================================
# data_integrity_check.py - Raw Bar Data Integrity Diagnostic
# =====================================================
"""
Standalone diagnostic for 3 specific symbols where valid_patterns.csv showed
entry_price deviating from open_price by > 25% on sub-50% gap stocks.

Checks per symbol/date:
  1. Does backtest open_price match the first raw bar's open?
  2. Does backtest gap_percent match (first_bar_open - prev_close) / prev_close * 100?
  3. Does entry_price appear anywhere in the day's bars, and at which timestamp?

Verdicts: DATA_OK | PRICE_MISMATCH | ENTRY_NOT_IN_BARS

Output: S:\trading\reports\data_integrity_<timestamp>\data_integrity_report.csv
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# ── Subjects ─────────────────────────────────────────────────────────────────
# Populate these from the rows in valid_patterns.csv before running.
SUBJECTS = [
    {'symbol': 'HOVR', 'date': '2024-01-04', 'open_price': 8.86,   'entry_price': 11.29, 'gap_percent': 10.47},
    {'symbol': 'RCAC', 'date': '2024-01-05', 'open_price': 6.95,   'entry_price': 10.77, 'gap_percent': 25.91},
    {'symbol': 'SPEC', 'date': '2024-01-08', 'open_price': 2.13,   'entry_price': 3.85,  'gap_percent': 18.99},
]

# ── Paths ─────────────────────────────────────────────────────────────────────
TICKER_DATA_DIR    = Path(r'S:\trading\ticker_data')
AGGREGATE_DIR      = Path(r'S:\trading\daily_aggregates')
REPORTS_ROOT       = Path(r'S:\trading\reports')

# Tolerance for floating-point price comparisons (0.5%)
PRICE_TOLERANCE    = 0.005
# Tolerance for scanning bars: entry_price considered "found" if any bar
# has open/high/low/close within this % of entry_price
ENTRY_SCAN_TOLERANCE = 0.001   # 0.1%


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_day_bars(symbol: str, date_str: str) -> pd.DataFrame:
    """Load raw minute bars for a symbol on a date from the parquet file."""
    date_obj = pd.Timestamp(date_str)
    year  = str(date_obj.year)
    month = f"{date_obj.month:02d}"
    filepath = TICKER_DATA_DIR / year / month / f"{date_str}.parquet"

    if not filepath.exists():
        return pd.DataFrame(), str(filepath)

    try:
        df = pd.read_parquet(filepath)
        df.columns = df.columns.str.lower()
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol].copy()
        if df.empty:
            return df, str(filepath)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df, str(filepath)
    except Exception as e:
        return pd.DataFrame(), f"{filepath} [ERROR: {e}]"


def _load_prev_close(symbol: str, date_str: str) -> tuple:
    """
    Find the most recent prior trading day's close from daily_aggregates.
    Returns (prev_close, prev_date_str) or (None, None).
    """
    date_obj = pd.Timestamp(date_str)

    for lookback in range(1, 8):
        prev_date = date_obj - pd.Timedelta(days=lookback)
        year_month = prev_date.strftime('%Y-%m')
        agg_file = AGGREGATE_DIR / str(prev_date.year) / f"{year_month}.parquet"

        if not agg_file.exists():
            continue

        try:
            agg = pd.read_parquet(agg_file)
            agg.columns = agg.columns.str.lower()
            if 'date' in agg.columns:
                agg['date'] = pd.to_datetime(agg['date']).dt.date
            row = agg[(agg['symbol'] == symbol) & (agg['date'] == prev_date.date())]
            if not row.empty:
                close_val = float(row.iloc[0]['close'])
                return close_val, prev_date.strftime('%Y-%m-%d')
        except Exception:
            continue

    return None, None


def _pct_diff(a: float, b: float) -> float:
    """Absolute percentage difference between two prices."""
    if b == 0:
        return float('inf')
    return abs(a - b) / abs(b) * 100.0


def _price_near(target: float, value: float, tol: float = ENTRY_SCAN_TOLERANCE) -> bool:
    return abs(target - value) / max(abs(value), 1e-9) <= tol


def _find_entry_in_bars(bars: pd.DataFrame, entry_price: float) -> tuple:
    """
    Scan all OHLC columns for the first bar that contains entry_price within tolerance.
    Returns (found: bool, timestamp_et: str | None, column: str | None).
    """
    if bars.empty or entry_price is None:
        return False, None, None

    for col in ('open', 'high', 'low', 'close'):
        if col not in bars.columns:
            continue
        mask = bars[col].apply(lambda v: _price_near(entry_price, float(v)))
        hits = bars[mask]
        if not hits.empty:
            ts = hits.iloc[0].get('timestamp', None)
            ts_str = None
            if ts is not None:
                ts_et = pd.Timestamp(ts).tz_convert('US/Eastern')
                ts_str = ts_et.strftime('%Y-%m-%d %H:%M ET')
            return True, ts_str, col

    return False, None, None


# ── Main diagnostic ───────────────────────────────────────────────────────────

def run_check(subject: dict) -> dict:
    symbol      = subject['symbol']
    date_str    = subject['date']
    bt_open     = subject.get('open_price')
    bt_entry    = subject.get('entry_price')
    bt_gap      = subject.get('gap_percent')

    result = {
        'symbol':                   symbol,
        'date':                     date_str,
        'bt_open_price':            bt_open,
        'bt_entry_price':           bt_entry,
        'bt_gap_percent':           bt_gap,
        # raw bar values
        'raw_first_bar_open':       None,
        'raw_first_bar_timestamp':  None,
        'raw_bar_count':            0,
        'raw_file_path':            None,
        # prev close
        'prev_close':               None,
        'prev_close_date':          None,
        # derived checks
        'open_price_match':         None,
        'open_pct_deviation':       None,
        'calc_gap_percent':         None,
        'gap_match':                None,
        'gap_deviation':            None,
        'entry_found_in_bars':      False,
        'entry_found_at_timestamp': None,
        'entry_found_in_column':    None,
        # verdict
        'verdict':                  'UNKNOWN',
        'verdict_detail':           '',
    }

    # ── Step 1: load raw bars ─────────────────────────────────────────────────
    bars, filepath = _load_day_bars(symbol, date_str)
    result['raw_file_path'] = filepath

    if bars.empty:
        result['verdict']        = 'NO_DATA'
        result['verdict_detail'] = f"No bars found at {filepath}"
        _print_verdict(result)
        return result

    result['raw_bar_count'] = len(bars)
    first_bar = bars.iloc[0]
    raw_open  = float(first_bar['open'])
    result['raw_first_bar_open'] = raw_open

    if 'timestamp' in first_bar.index and pd.notna(first_bar['timestamp']):
        ts_et = pd.Timestamp(first_bar['timestamp']).tz_convert('US/Eastern')
        result['raw_first_bar_timestamp'] = ts_et.strftime('%Y-%m-%d %H:%M ET')

    # ── Step 2: prev close from aggregates ───────────────────────────────────
    prev_close, prev_date = _load_prev_close(symbol, date_str)
    result['prev_close']      = prev_close
    result['prev_close_date'] = prev_date

    # ── Check A: open_price vs first bar open ─────────────────────────────────
    issues = []

    if bt_open is not None:
        open_dev = _pct_diff(float(bt_open), raw_open)
        result['open_pct_deviation'] = round(open_dev, 3)
        result['open_price_match']   = open_dev <= (PRICE_TOLERANCE * 100)
        if not result['open_price_match']:
            issues.append(
                f"open_mismatch: bt={bt_open:.4f} raw={raw_open:.4f} dev={open_dev:.2f}%"
            )
    else:
        result['open_price_match'] = None   # can't check — subject value not provided

    # ── Check B: gap_percent vs calculated from raw ───────────────────────────
    if prev_close is not None and prev_close > 0:
        calc_gap = (raw_open - prev_close) / prev_close * 100.0
        result['calc_gap_percent'] = round(calc_gap, 4)

        if bt_gap is not None:
            gap_dev = abs(float(bt_gap) - calc_gap)
            result['gap_deviation']  = round(gap_dev, 4)
            result['gap_match']      = gap_dev <= 0.5   # within 0.5 percentage points
            if not result['gap_match']:
                issues.append(
                    f"gap_mismatch: bt={bt_gap:.2f}% calc={calc_gap:.2f}% "
                    f"(prev_close={prev_close:.4f} on {prev_date})"
                )
        else:
            result['gap_match'] = None
    else:
        issues.append(f"prev_close_missing: could not find prior close within 7 days of {date_str}")

    # ── Check C: entry_price somewhere in bars ────────────────────────────────
    if bt_entry is not None:
        found, entry_ts, entry_col = _find_entry_in_bars(bars, float(bt_entry))
        result['entry_found_in_bars']      = found
        result['entry_found_at_timestamp'] = entry_ts
        result['entry_found_in_column']    = entry_col
        if not found:
            issues.append(
                f"entry_not_in_bars: entry_price={bt_entry:.4f} not found "
                f"in any OHLC column across {len(bars)} bars "
                f"(bar range open={bars['open'].min():.4f}–{bars['open'].max():.4f})"
            )

    # ── Verdict ───────────────────────────────────────────────────────────────
    if not issues:
        result['verdict']        = 'DATA_OK'
        result['verdict_detail'] = 'All checks passed'
    elif any('entry_not_in_bars' in i for i in issues) and any('mismatch' in i for i in issues):
        result['verdict']        = 'PRICE_MISMATCH'
        result['verdict_detail'] = ' | '.join(issues)
    elif any('mismatch' in i for i in issues):
        result['verdict']        = 'PRICE_MISMATCH'
        result['verdict_detail'] = ' | '.join(issues)
    else:
        result['verdict']        = 'ENTRY_NOT_IN_BARS'
        result['verdict_detail'] = ' | '.join(issues)

    _print_verdict(result)
    return result


def _print_verdict(r: dict):
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  {r['symbol']}  |  {r['date']}  |  verdict: {r['verdict']}")
    print(sep)
    print(f"  Raw file           : {r['raw_file_path']}")
    print(f"  Raw bar count      : {r['raw_bar_count']}")
    print(f"  Raw first bar open : {r['raw_first_bar_open']}  @ {r['raw_first_bar_timestamp']}")
    print(f"  Backtest open      : {r['bt_open_price']}  (deviation: {r['open_pct_deviation']}%  match={r['open_price_match']})")
    print(f"  Prev close         : {r['prev_close']}  ({r['prev_close_date']})")
    print(f"  Calc gap           : {r['calc_gap_percent']}%  |  BT gap: {r['bt_gap_percent']}%  (match={r['gap_match']})")
    print(f"  Entry price        : {r['bt_entry_price']}")
    print(f"  Entry in bars      : {r['entry_found_in_bars']}  col={r['entry_found_in_column']}  @ {r['entry_found_at_timestamp']}")
    if r['verdict_detail'] and r['verdict'] != 'DATA_OK':
        print(f"  Issues:")
        for part in r['verdict_detail'].split(' | '):
            print(f"    • {part}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Allow subjects to be overridden via CLI:
    #    python data_integrity_check.py HOVR:2024-01-10:open:entry:gap RCAC:... SPEC:...
    if len(sys.argv) > 1:
        SUBJECTS.clear()
        for arg in sys.argv[1:]:
            parts = arg.split(':')
            if len(parts) >= 2:
                SUBJECTS.append({
                    'symbol':      parts[0].upper(),
                    'date':        parts[1],
                    'open_price':  float(parts[2]) if len(parts) > 2 and parts[2] else None,
                    'entry_price': float(parts[3]) if len(parts) > 3 and parts[3] else None,
                    'gap_percent': float(parts[4]) if len(parts) > 4 and parts[4] else None,
                })

    print("\n" + "=" * 70)
    print("  DATA INTEGRITY CHECK")
    print(f"  Subjects : {[s['symbol'] for s in SUBJECTS]}")
    print(f"  Data dir : {TICKER_DATA_DIR}")
    print(f"  Agg dir  : {AGGREGATE_DIR}")
    print("=" * 70)

    records = []
    for subject in SUBJECTS:
        records.append(run_check(subject))

    # ── Write report ─────────────────────────────────────────────────────────
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = REPORTS_ROOT / f"data_integrity_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "data_integrity_report.csv"
    pd.DataFrame(records).to_csv(report_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"  Report  : {report_path}")
    print(f"  Records : {len(records)}")
    verdicts = [r['verdict'] for r in records]
    for v in sorted(set(verdicts)):
        print(f"    {v:<22} × {verdicts.count(v)}")
    print("=" * 70)