"""Quick test to verify gap calculator is filtering zero prices."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Force reload to pick up changes
import importlib
import data_handler.gap_calculator
importlib.reload(data_handler.gap_calculator)

from config.loader import build_config
from data_handler.local import LocalDataHandler
import pandas as pd

cfg = build_config()
dh = LocalDataHandler(cfg, data_dir=cfg.backtest.DATA_DIR)

# Test on 2024-01-03 (the problematic date)
day = pd.Timestamp("2024-01-03")
res = dh.calculate_gaps(day, premarket=True)
gaps = res.get("gaps", pd.DataFrame())

print("=" * 80)
print("GAP CALCULATION VERIFICATION")
print("=" * 80)
print(f"Date: {day.date()}")
print(f"Previous day used: {res.get('prev_day_used')}")
print(f"Lookback days: {res.get('lookback_days')}")
print(f"Total symbols: {len(gaps)}")
print()

# ✅ NEW: Check symbol sets
prev_syms = res.get('prev_syms', set())
today_syms = res.get('today_syms', set())
only_prev = prev_syms - today_syms  # Have prev close but missing today open
only_today = today_syms - prev_syms  # Have today open but missing prev close

print("SYMBOL SET ANALYSIS:")
print(f"  • Symbols with prev close: {len(prev_syms)}")
print(f"  • Symbols with today open: {len(today_syms)}")
print(f"  • Common symbols (in gaps): {len(gaps)}")
print(f"  • Missing today open (no_open_bar): {len(only_prev)}")
print(f"  • Missing prev close (no_prev_close): {len(only_today)}")
print()

if len(only_prev) > 0:
    print("Sample symbols with no_open_bar (have prev close but missing today open):")
    print(f"  {', '.join(list(only_prev)[:10])}")
    print()

if len(only_today) > 0:
    print("Sample symbols with no_prev_close (have today open but missing prev close):")
    print(f"  {', '.join(list(only_today)[:10])}")
    print()

if not gaps.empty:
    # Check for any remaining zero prices
    zero_prev_close = (gaps['prev_close'] == 0).sum()
    zero_open = (gaps['open_price'] == 0).sum()
    
    print("ZERO PRICE CHECK:")
    print(f"  • Symbols with prev_close=0: {zero_prev_close}")
    print(f"  • Symbols with open_price=0: {zero_open}")
    
    if zero_prev_close > 0 or zero_open > 0:
        print()
        print("❌ STILL HAVE ZERO PRICES - Fix not working!")
        print()
        print("Symbols with zero prices:")
        zero_mask = (gaps['prev_close'] == 0) | (gaps['open_price'] == 0)
        print(gaps[zero_mask][['symbol', 'open_price', 'prev_close', 'gap_percent']])
    else:
        print()
        print("✓ No zero prices found - Fix is working!")
        print()
        print("Gap statistics:")
        print(f"  • Min gap: {gaps['gap_percent'].min():.2f}%")
        print(f"  • Max gap: {gaps['gap_percent'].max():.2f}%")
        print(f"  • Median gap: {gaps['gap_percent'].median():.2f}%")
        
        # Check for inf values
        inf_count = (gaps['gap_percent'] == float('inf')).sum()
        if inf_count > 0:
            print(f"  • ⚠️  WARNING: {inf_count} infinite gap percentages!")
        
        print()
        print("Top 10 gaps:")
        top_10 = gaps.nlargest(10, 'gap_percent')[['symbol', 'open_price', 'prev_close', 'gap_percent']]
        for _, row in top_10.iterrows():
            print(f"  {row['symbol']}: {row['gap_percent']:+.2f}% (open=${row['open_price']:.2f}, prev_close=${row['prev_close']:.2f})")
else:
    print("⚠️  No gaps calculated - check if data exists")

print("=" * 80)