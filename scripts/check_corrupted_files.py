"""
Standalone diagnostic script to check parquet files - no imports from project.
"""

import pandas as pd
from pathlib import Path
import re

def main():
    """Check all parquet files for corruption."""
    
    data_dir = Path(r"D:\trading_data")
    pattern = re.compile(r"^(\d{8})\.parquet$", re.IGNORECASE)
    
    print("=" * 80)
    print("PARQUET FILE CORRUPTION CHECK")
    print("=" * 80)
    print(f"\nScanning: {data_dir}\n")
    
    total_files = 0
    valid_files = 0
    corrupted_files = []
    empty_files = []
    missing_symbol = []
    
    if not data_dir.exists():
        print(f"❌ ERROR: Directory does not exist: {data_dir}")
        return
    
    # Scan all files
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
                
                total_files += 1
                date_compact = match.group(1)
                date_str = f"{date_compact[0:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
                
                try:
                    # Try to read the file
                    df = pd.read_parquet(file)
                    
                    # Check if empty
                    if df.empty:
                        empty_files.append((date_str, str(file)))
                        print(f"❌ {date_str}: EMPTY - {file.name}")
                        continue
                    
                    # Check for symbol column
                    if 'symbol' not in df.columns:
                        missing_symbol.append((date_str, str(file), list(df.columns)))
                        print(f"❌ {date_str}: NO 'symbol' COLUMN - {file.name}")
                        print(f"   Columns: {list(df.columns)}")
                        continue
                    
                    # File is valid
                    valid_files += 1
                    symbols = len(df['symbol'].unique())
                    rows = len(df)
                    
                    # Only print every 10th valid file to reduce noise
                    if valid_files % 10 == 0:
                        print(f"✓ {date_str}: {symbols} symbols, {rows} rows")
                    
                except Exception as e:
                    corrupted_files.append((date_str, str(file), str(e)))
                    print(f"❌ {date_str}: CORRUPTED - {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files scanned: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Empty files: {len(empty_files)}")
    print(f"Missing 'symbol' column: {len(missing_symbol)}")
    
    # Show corrupted files details
    if corrupted_files:
        print("\n" + "=" * 80)
        print("CORRUPTED FILES DETAILS")
        print("=" * 80)
        for date_str, path, error in corrupted_files:
            print(f"\nDate: {date_str}")
            print(f"Path: {path}")
            print(f"Error: {error}")
    
    # Export corrupted list
    if corrupted_files:
        output_file = Path(__file__).parent.parent / "corrupted_files.txt"
        with open(output_file, 'w') as f:
            f.write("CORRUPTED PARQUET FILES\n")
            f.write("=" * 80 + "\n\n")
            for date_str, path, error in corrupted_files:
                f.write(f"{date_str}\t{path}\t{error}\n")
        
        print(f"\n✓ Corrupted files list saved to: {output_file}")
        print("\nTO FIX: Delete these files and re-download the data for these dates")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
