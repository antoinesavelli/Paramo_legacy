# -*- coding: utf-8 -*-
"""
Diagnostic to find symbols with all-zero OHLC prices.
"""

import sys  
import os
import pandas as pd
from pathlib import Path

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config import TradingConfig

def diagnose_zero_prices():
    """Find symbols with all-zero or mostly-zero prices."""
    
    config = TradingConfig()
    data_dir = Path(config.backtest.DATA_DIR)
    
    print("=" * 80)
    print("ZERO PRICE DIAGNOSTIC")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    
    # Check 2024-01-03
    test_date = "2024-01-03"
    year = "2024"
    month = "01"
    date_compact = "20240103"
    
    filepath = data_dir / year / month / f"{date_compact}.parquet"
    
    if not filepath.exists():
        print(f"\n❌ File not found: {filepath}")
        return
    
    print(f"\n✓ Reading file: {filepath}")
    df = pd.read_parquet(filepath)
    
    print(f"Total rows: {len(df):,}")
    print(f"Total symbols: {df['symbol'].nunique()}")
    
    # Analyze zero prices
    print("\n" + "=" * 80)
    print("ANALYZING ZERO PRICES")
    print("=" * 80)
    
    zero_open_symbols = []
    zero_close_symbols = []
    all_zero_ohlc_symbols = []
    partial_zero_symbols = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        
        # Count zeros
        zero_opens = (symbol_df['open'] == 0).sum()
        zero_closes = (symbol_df['close'] == 0).sum()
        zero_highs = (symbol_df['high'] == 0).sum()
        zero_lows = (symbol_df['low'] == 0).sum()
        
        total_bars = len(symbol_df)
        zero_pct = (zero_opens / total_bars) * 100 if total_bars > 0 else 0
        
        # Categorize
        if zero_opens == total_bars:
            zero_open_symbols.append({
                'symbol': symbol,
                'bars': total_bars,
                'zero_opens': zero_opens,
                'zero_closes': zero_closes,
                'first_bar': symbol_df.iloc[0]['open'] if total_bars > 0 else None,
                'last_bar': symbol_df.iloc[-1]['close'] if total_bars > 0 else None
            })
        
        if zero_closes == total_bars:
            zero_close_symbols.append(symbol)
        
        if (zero_opens == total_bars and zero_closes == total_bars and 
            zero_highs == total_bars and zero_lows == total_bars):
            all_zero_ohlc_symbols.append(symbol)
        
        if zero_pct > 0 and zero_pct < 100:
            partial_zero_symbols.append({
                'symbol': symbol,
                'bars': total_bars,
                'zero_opens': zero_opens,
                'zero_pct': zero_pct
            })
    
    # Report
    print(f"\nSymbols with ALL bars having zero open: {len(zero_open_symbols)}")
    if zero_open_symbols:
        print("\nTop 20 symbols with zero opens:")
        for item in zero_open_symbols[:20]:
            print(f"  • {item['symbol']}: {item['bars']} bars, "
                  f"first={item['first_bar']}, last={item['last_bar']}")
    
    print(f"\nSymbols with ALL OHLC values zero: {len(all_zero_ohlc_symbols)}")
    if all_zero_ohlc_symbols:
        print(f"  Examples: {all_zero_ohlc_symbols[:10]}")
    
    print(f"\nSymbols with PARTIAL zero opens: {len(partial_zero_symbols)}")
    if partial_zero_symbols:
        print("\nTop 10 symbols with partial zeros:")
        sorted_partial = sorted(partial_zero_symbols, key=lambda x: x['zero_pct'], reverse=True)
        for item in sorted_partial[:10]:
            print(f"  • {item['symbol']}: {item['zero_opens']}/{item['bars']} bars "
                  f"({item['zero_pct']:.1f}%) have zero open")
    
    # Check what the gap calculator would see
    print("\n" + "=" * 80)
    print("GAP CALCULATOR PERSPECTIVE")
    print("=" * 80)
    
    # Simulate what gap calc does
    td = df[['symbol', 'timestamp', 'open', 'close']].dropna(subset=['symbol', 'timestamp', 'open']).copy()
    print(f"\nAfter dropna: {len(td)} rows, {td['symbol'].nunique()} symbols")
    
    # Filter zero opens (what gap calc does)
    td_filtered = td[td['open'] > 0]
    print(f"After filtering open > 0: {len(td_filtered)} rows, {td_filtered['symbol'].nunique()} symbols")
    
    # Which symbols were completely removed?
    original_symbols = set(df['symbol'].unique())
    remaining_symbols = set(td_filtered['symbol'].unique())
    removed_symbols = original_symbols - remaining_symbols
    
    print(f"\nSymbols REMOVED by zero-open filter: {len(removed_symbols)}")
    if removed_symbols:
        print(f"  Examples: {sorted(list(removed_symbols))[:20]}")
    
    # Show a specific problematic symbol
    if zero_open_symbols:
        problem_symbol = zero_open_symbols[0]['symbol']
        print(f"\n" + "=" * 80)
        print(f"DETAILED ANALYSIS: {problem_symbol}")
        print("=" * 80)
        
        sym_df = df[df['symbol'] == problem_symbol]
        print(f"\nAll bars for {problem_symbol}:")
        print(sym_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head(20).to_string(index=False))
        
        print(f"\nStatistics:")
        print(f"  Open: min={sym_df['open'].min()}, max={sym_df['open'].max()}, "
              f"mean={sym_df['open'].mean():.4f}")
        print(f"  Close: min={sym_df['close'].min()}, max={sym_df['close'].max()}, "
              f"mean={sym_df['close'].mean():.4f}")
        print(f"  Volume: total={sym_df['volume'].sum():,}, "
              f"mean={sym_df['volume'].mean():.0f}")

if __name__ == "__main__":
    try:
        diagnose_zero_prices()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    input("\nPress Enter to exit...")
