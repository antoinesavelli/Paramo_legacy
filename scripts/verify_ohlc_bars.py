# Diagnostic script to check data availability and timing
import pandas as pd
from pathlib import Path

data_dir = Path(r"D:\trading_data")

print("=" * 80)
print("DATA AVAILABILITY CHECK")
print("=" * 80)

# Check if data directory exists
if not data_dir.exists():
    print(f"ERROR: Data directory does not exist: {data_dir}")
    input("Press Enter to exit...")
    exit(1)

print(f"Data directory: {data_dir}")
print()

# Check the structure for January 2024
print("Checking for January 2024 data...")
print()

# Try both month formats
jan_paths = [
    data_dir / "2024" / "01",  # Zero-padded
    data_dir / "2024" / "1",   # Non-padded
]

files_found = []
for path in jan_paths:
    if path.exists():
        print(f"✓ Found directory: {path}")
        files = list(path.glob("*.parquet"))
        if files:
            print(f"  Contains {len(files)} parquet files")
            files_found.extend(files)
            # Show first 5 files
            for f in sorted(files)[:5]:
                print(f"    - {f.name}")
        else:
            print(f"  WARNING: Directory exists but no parquet files found")
    else:
        print(f"✗ Not found: {path}")

print()

if not files_found:
    print("ERROR: No data files found for January 2024")
    print("\nSearching entire data directory...")
    all_files = list(data_dir.rglob("*.parquet"))
    print(f"Total parquet files found: {len(all_files)}")
    
    if all_files:
        print("\nDirectory structure:")
        years = {}
        for f in all_files:
            year = f.parent.parent.name
            month = f.parent.name
            if year not in years:
                years[year] = set()
            years[year].add(month)
        
        for year in sorted(years.keys()):
            months = sorted(years[year])
            print(f"  {year}/: {', '.join(months)}")
    
    input("\nPress Enter to exit...")
    exit(1)

print("=" * 80)
print("TIMESTAMP ANALYSIS FOR 2024-01-02")
print("=" * 80)

# Try to find the specific file for backtest start date
target_files = [
    data_dir / "2024" / "01" / "20240102.parquet",
    data_dir / "2024" / "1" / "20240102.parquet",
]

sample_file = None
for f in target_files:
    if f.exists():
        sample_file = f
        print(f"✓ Found: {f}")
        break

if not sample_file:
    # Use first available file from January
    sample_file = sorted(files_found)[0]
    print(f"Using first available file: {sample_file}")

print()

try:
    df = pd.read_parquet(sample_file)
    
    print(f"Total rows: {len(df):,}")
    print(f"Total symbols: {df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'}")
    print()
    
    # Check timestamp format
    if 'timestamp' not in df.columns:
        print("ERROR: No 'timestamp' column found!")
        print(f"Available columns: {', '.join(df.columns)}")
        input("Press Enter to exit...")
        exit(1)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Timestamp Info:")
    print(f"  - Min: {df['timestamp'].min()}")
    print(f"  - Max: {df['timestamp'].max()}")
    print(f"  - Timezone: {df['timestamp'].dt.tz}")
    print()
    
    # Time distribution
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    print("Time coverage:")
    hours = sorted(df['hour'].unique())
    print(f"  - Hours: {hours[0]}:00 to {hours[-1]}:00")
    print()
    
    # Check critical times
    print("Critical time points:")
    
    critical_times = [
        ('4:00 AM', 4, 0),
        ('9:30 AM', 9, 30),
        ('3:59 PM', 15, 59),
        ('4:00 PM', 16, 0),
    ]
    
    all_symbols = df['symbol'].unique()
    total_symbols = len(all_symbols)
    
    for label, hour, minute in critical_times:
        mask = (df['hour'] == hour) & (df['minute'] == minute)
        count = mask.sum()
        symbols = df[mask]['symbol'].nunique()
        pct = (symbols / total_symbols * 100) if total_symbols > 0 else 0
        print(f"  - {label}: {count:,} bars, {symbols}/{total_symbols} symbols ({pct:.1f}%)")
    
    print()
    
    # Check last bar distribution
    print("Last bar time per symbol:")
    last_times = df.groupby('symbol')['timestamp'].max()
    last_time_str = last_times.dt.strftime('%H:%M')
    top_times = last_time_str.value_counts().head(5)
    
    for time_str, count in top_times.items():
        pct = (count / total_symbols * 100)
        print(f"  - {time_str}: {count} symbols ({pct:.1f}%)")
    
    print()
    
    # Diagnosis
    symbols_at_4pm = df[(df['hour'] == 16) & (df['minute'] == 0)]['symbol'].nunique()
    symbols_at_359pm = df[(df['hour'] == 15) & (df['minute'] == 59)]['symbol'].nunique()
    
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if symbols_at_4pm < total_symbols * 0.5:
        print(f"\n*** ISSUE: Only {symbols_at_4pm}/{total_symbols} symbols ({symbols_at_4pm/total_symbols*100:.1f}%) have 4:00 PM bars ***")
        print(f"    But {symbols_at_359pm}/{total_symbols} symbols ({symbols_at_359pm/total_symbols*100:.1f}%) have 3:59 PM bars")
        print("\nROOT CAUSE: Market close data is at 3:59 PM, not 4:00 PM")
        print("\nSOLUTION: Update gap_calculator.py to use <= 3:59 PM or use LAST bar of day")
    else:
        print(f"\n✓ Good: {symbols_at_4pm}/{total_symbols} symbols have 4:00 PM bars")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
input("\nPress Enter to exit...")