"""
Diagnostic script to find bars with zero prices in parquet files.
This is the root cause of infinite gap percentages.
"""

import sys
from pathlib import Path
import pandas as pd
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.loader import build_config

def analyze_zero_prices(data_dir: Path) -> dict:
    """Scan all parquet files for bars with zero prices."""
    pattern = re.compile(r"^(\d{8})\.parquet$", re.IGNORECASE)
    
    report = {
        'total_files': 0,
        'files_scanned': 0,
        'files_with_zero_prices': 0,
        'total_zero_open': 0,
        'total_zero_high': 0,
        'total_zero_low': 0,
        'total_zero_close': 0,
        'symbols_with_zero_close': set(),
        'dates_with_zero_close': [],
        'critical_last_bar_issues': []  # Files where LAST bar has close=0
    }
    
    print("=" * 80)
    print("SCANNING FOR ZERO-PRICE BARS")
    print("=" * 80)
    print()
    
    for year_dir in data_dir.iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit() or len(year_dir.name) != 4:
            continue
            
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir() or not month_dir.name.isdigit() or len(month_dir.name) != 2:
                continue
            
            for file in month_dir.glob("*.parquet"):
                match = pattern.match(file.name)
                if not match:
                    continue
                
                report['total_files'] += 1
                date_compact = match.group(1)
                date_str = f"{date_compact[0:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
                
                try:
                    df = pd.read_parquet(file)
                    report['files_scanned'] += 1
                    
                    if df.empty:
                        continue
                    
                    # Check for required columns
                    if not all(col in df.columns for col in ['symbol', 'open', 'high', 'low', 'close']):
                        continue
                    
                    # Count zero prices
                    zero_open = (df['open'] == 0).sum()
                    zero_high = (df['high'] == 0).sum()
                    zero_low = (df['low'] == 0).sum()
                    zero_close = (df['close'] == 0).sum()
                    
                    has_zeros = zero_open > 0 or zero_high > 0 or zero_low > 0 or zero_close > 0
                    
                    if has_zeros:
                        report['files_with_zero_prices'] += 1
                        report['total_zero_open'] += zero_open
                        report['total_zero_high'] += zero_high
                        report['total_zero_low'] += zero_low
                        report['total_zero_close'] += zero_close
                        
                        # Find symbols with zero close
                        symbols_zero_close = df[df['close'] == 0]['symbol'].unique()
                        report['symbols_with_zero_close'].update(symbols_zero_close)
                        
                        if zero_close > 0:
                            report['dates_with_zero_close'].append(date_str)
                        
                        print(f"❌ {date_str}:")
                        print(f"   Zero prices found: open={zero_open}, high={zero_high}, low={zero_low}, close={zero_close}")
                        if len(symbols_zero_close) > 0:
                            print(f"   Symbols with close=0: {', '.join(list(symbols_zero_close)[:10])}")
                        
                        # CRITICAL: Check if LAST bar has zero close (breaks gap calculation)
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df_sorted = df.sort_values('timestamp')
                            last_bars = df_sorted.groupby('symbol').tail(1)
                            last_bar_zero_close = last_bars[last_bars['close'] == 0]
                            
                            if not last_bar_zero_close.empty:
                                affected_symbols = last_bar_zero_close['symbol'].tolist()
                                report['critical_last_bar_issues'].append({
                                    'date': date_str,
                                    'symbols': affected_symbols,
                                    'count': len(affected_symbols)
                                })
                                print(f"   ⚠️  CRITICAL: {len(affected_symbols)} symbols have LAST BAR with close=0!")
                                print(f"      This breaks gap calculations: {', '.join(affected_symbols[:5])}")
                        print()
                    
                except Exception as e:
                    print(f"❌ Error reading {date_str}: {e}")
    
    return report

def main():
    """Run zero-price diagnostics."""
    print("=" * 80)
    print("ZERO-PRICE BAR DIAGNOSTICS")
    print("=" * 80)
    print()
    
    # Load config
    config = build_config()
    data_dir = Path(config.backtest.DATA_DIR)
    
    print(f"Data directory: {data_dir}")
    print()
    
    # Scan for zero prices
    report = analyze_zero_prices(data_dir)
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total files: {report['total_files']}")
    print(f"Files scanned: {report['files_scanned']}")
    print(f"Files with zero prices: {report['files_with_zero_prices']}")
    print()
    print(f"Total bars with zero prices:")
    print(f"  • Zero open:  {report['total_zero_open']:,}")
    print(f"  • Zero high:  {report['total_zero_high']:,}")
    print(f"  • Zero low:   {report['total_zero_low']:,}")
    print(f"  • Zero close: {report['total_zero_close']:,}")
    print()
    print(f"Unique symbols with close=0: {len(report['symbols_with_zero_close'])}")
    print(f"Dates with zero close prices: {len(report['dates_with_zero_close'])}")
    
    if report['critical_last_bar_issues']:
        print()
        print("=" * 80)
        print("⚠️  CRITICAL ISSUE: LAST BAR HAS CLOSE=0")
        print("=" * 80)
        print()
        print(f"Found {len(report['critical_last_bar_issues'])} dates where symbols have")
        print("their LAST bar with close=0. This causes infinite gap percentages!")
        print()
        print("Affected dates:")
        for issue in report['critical_last_bar_issues'][:10]:
            print(f"  • {issue['date']}: {issue['count']} symbols")
            print(f"    Examples: {', '.join(issue['symbols'][:5])}")
        
        if len(report['critical_last_bar_issues']) > 10:
            print(f"  ... and {len(report['critical_last_bar_issues']) - 10} more dates")
    
    # Export detailed report
    import json
    report_file = project_root / "zero_price_report.json"
    
    serializable_report = {
        'total_files': report['total_files'],
        'files_scanned': report['files_scanned'],
        'files_with_zero_prices': report['files_with_zero_prices'],
        'total_zero_open': report['total_zero_open'],
        'total_zero_high': report['total_zero_high'],
        'total_zero_low': report['total_zero_low'],
        'total_zero_close': report['total_zero_close'],
        'symbols_with_zero_close_count': len(report['symbols_with_zero_close']),
        'symbols_with_zero_close': sorted(list(report['symbols_with_zero_close'])),
        'dates_with_zero_close': report['dates_with_zero_close'],
        'critical_last_bar_issues': report['critical_last_bar_issues']
    }
    
    with open(report_file, 'w') as f:
        json.dump(serializable_report, f, indent=2)
    
    print()
    print(f"✓ Full report exported to: {report_file}")
    
    # Recommendations
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    if report['total_zero_close'] == 0:
        print("✓ No zero-price bars found - data quality is excellent!")
    else:
        print("⚠️  ACTION REQUIRED:")
        print()
        print("1. IMMEDIATE FIX - Update gap_calculator.py:")
        print("   Add filtering to remove zero prices:")
        print()
        print("   In calculate_gaps() method:")
        print("   • Line ~128: pv = pv[pv['close'] > 0]")
        print("   • Line ~111: td = td[td['open'] > 0]")
        print()
        print("   In _calculate_symbol_gap_simple() method:")
        print("   • After loading prev_df: prev_df = prev_df[prev_df['close'] > 0]")
        print("   • After loading today_df: today_df = today_df[today_df['open'] > 0]")
        print()
        print("2. DATA CLEANING (long-term):")
        print(f"   • {report['files_with_zero_prices']} files have zero-price bars")
        print("   • Consider re-downloading or cleaning these files")
        print("   • Or implement data validation at ingestion time")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()