"""
Diagnostic script to examine the structure of aggregate parquet files.
This helps understand the schema, index structure, and data organization.
"""

import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.loader import build_config


def examine_parquet_structure(file_path: Path) -> dict:
    """
    Examine a parquet file's structure in detail.
    
    Returns detailed information about schema, index, data types, and sample data.
    """
    info = {
        'file_path': str(file_path),
        'file_size_mb': file_path.stat().st_size / (1024 * 1024),
        'exists': file_path.exists()
    }
    
    if not file_path.exists():
        print(f"❌ File does not exist: {file_path}")
        return info
    
    try:
        # Read parquet metadata using PyArrow
        parquet_file = pq.read_table(file_path)
        
        print("=" * 80)
        print(f"FILE: {file_path.name}")
        print("=" * 80)
        print()
        
        # File size
        print(f"📊 File Size: {info['file_size_mb']:.2f} MB")
        print()
        
        # PyArrow schema
        print("🔍 PARQUET SCHEMA (PyArrow):")
        print("-" * 80)
        print(parquet_file.schema)
        print()
        
        # Read with pandas to examine DataFrame structure
        df = pd.read_parquet(file_path)
        
        # DataFrame info
        print("📋 DATAFRAME INFO:")
        print("-" * 80)
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        print()
        
        # Index information
        print("🔑 INDEX STRUCTURE:")
        print("-" * 80)
        print(f"Index type: {type(df.index).__name__}")
        print(f"Index names: {df.index.names}")
        
        if isinstance(df.index, pd.MultiIndex):
            print(f"Index levels: {df.index.nlevels}")
            for i, level_name in enumerate(df.index.names):
                level_values = df.index.get_level_values(i)
                print(f"  Level {i} ({level_name}):")
                print(f"    - dtype: {level_values.dtype}")
                print(f"    - unique values: {level_values.nunique():,}")
                print(f"    - sample: {list(level_values.unique()[:5])}")
        else:
            print(f"  - dtype: {df.index.dtype}")
            print(f"  - unique values: {df.index.nunique():,}")
            print(f"  - sample: {list(df.index.unique()[:5])}")
        print()
        
        # Column information
        print("📊 COLUMNS:")
        print("-" * 80)
        for col in df.columns:
            col_data = df[col]
            print(f"  {col}:")
            print(f"    - dtype: {col_data.dtype}")
            print(f"    - non-null: {col_data.notna().sum():,} / {len(col_data):,}")
            print(f"    - null count: {col_data.isna().sum():,}")
            
            if pd.api.types.is_numeric_dtype(col_data):
                print(f"    - min: {col_data.min()}")
                print(f"    - max: {col_data.max()}")
                print(f"    - mean: {col_data.mean():.4f}")
                
                # Check for zeros
                zero_count = (col_data == 0).sum()
                if zero_count > 0:
                    print(f"    - ⚠️  zeros: {zero_count:,} ({zero_count/len(col_data)*100:.2f}%)")
            else:
                unique_count = col_data.nunique()
                print(f"    - unique values: {unique_count:,}")
                if unique_count <= 10:
                    print(f"    - values: {col_data.unique().tolist()}")
        print()
        
        # Sample data
        print("📄 SAMPLE DATA (first 10 rows):")
        print("-" * 80)
        print(df.head(10).to_string())
        print()
        
        # If multi-index with 'date', show date range
        if isinstance(df.index, pd.MultiIndex) and 'date' in df.index.names:
            dates = df.index.get_level_values('date').unique()
            print(f"📅 DATE RANGE: {dates.min()} to {dates.max()} ({len(dates)} unique dates)")
            print()
        
        # If 'symbol' column or index level exists, show symbol info
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()
            print(f"🏢 SYMBOLS: {len(symbols):,} unique symbols")
            print(f"   Sample: {list(symbols[:10])}")
            print()
        elif isinstance(df.index, pd.MultiIndex) and 'symbol' in df.index.names:
            symbols = df.index.get_level_values('symbol').unique()
            print(f"🏢 SYMBOLS: {len(symbols):,} unique symbols")
            print(f"   Sample: {list(symbols[:10])}")
            print()
        
        # Check for data quality issues
        print("🔍 DATA QUALITY CHECKS:")
        print("-" * 80)
        
        # Check for duplicate index
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            print(f"⚠️  DUPLICATE INDEX: {dup_count:,} duplicate entries found")
        else:
            print("✅ No duplicate index entries")
        
        # Check for missing values
        missing = df.isna().sum()
        if missing.any():
            print(f"⚠️  MISSING VALUES:")
            for col, count in missing[missing > 0].items():
                print(f"   {col}: {count:,} ({count/len(df)*100:.2f}%)")
        else:
            print("✅ No missing values")
        
        # Check for zero prices in OHLC columns
        ohlc_cols = ['open', 'high', 'low', 'close']
        zero_prices = {}
        for col in ohlc_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    zero_prices[col] = zero_count
        
        if zero_prices:
            print(f"⚠️  ZERO PRICES FOUND:")
            for col, count in zero_prices.items():
                print(f"   {col}: {count:,} ({count/len(df)*100:.2f}%)")
        else:
            print("✅ No zero prices in OHLC columns")
        
        print()
        
        info.update({
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'index_type': type(df.index).__name__,
            'index_names': df.index.names,
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'has_duplicates': df.index.duplicated().any(),
            'missing_values': missing.to_dict(),
            'zero_prices': zero_prices
        })
        
    except Exception as e:
        print(f"❌ Error examining file: {e}")
        import traceback
        traceback.print_exc()
        info['error'] = str(e)
    
    return info


def scan_aggregate_directory(aggregate_dir: Path, max_files: int = 3):
    """
    Scan aggregate directory and examine structure of sample files.
    
    Args:
        aggregate_dir: Path to trading_data/daily_aggregates
        max_files: Maximum number of files to examine in detail
    """
    print("=" * 80)
    print("AGGREGATE FILE STRUCTURE EXAMINATION")
    print("=" * 80)
    print()
    print(f"📁 Aggregate directory: {aggregate_dir}")
    print()
    
    if not aggregate_dir.exists():
        print(f"❌ Directory does not exist: {aggregate_dir}")
        return
    
    # Find all parquet files
    parquet_files = []
    for year_dir in sorted(aggregate_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for file in sorted(year_dir.glob("*.parquet")):
            parquet_files.append(file)
    
    if not parquet_files:
        print("❌ No parquet files found in aggregate directory")
        return
    
    print(f"📊 Found {len(parquet_files)} aggregate files")
    print()
    
    # Examine a sample of files
    print(f"Examining {min(max_files, len(parquet_files))} sample files...")
    print()
    
    for i, file_path in enumerate(parquet_files[:max_files], 1):
        print(f"\n{'=' * 80}")
        print(f"SAMPLE {i}/{min(max_files, len(parquet_files))}")
        print(f"{'=' * 80}\n")
        
        examine_parquet_structure(file_path)
        
        if i < min(max_files, len(parquet_files)):
            input("\nPress Enter to examine next file...")


def main():
    """Run aggregate file structure examination."""
    print("=" * 80)
    print("PARQUET FILE STRUCTURE ANALYZER")
    print("=" * 80)
    print()
    
    # Load config
    config = build_config()
    
    # Point to aggregate files
    # Path: D:\trading_data\daily_aggregates\YYYY\YYYYMM.parquet
    aggregate_dir = Path(config.backtest.BASE_DATA_DIR) / "daily_aggregates"
    
    print(f"Configuration loaded:")
    print(f"  Base data dir: {config.backtest.BASE_DATA_DIR}")
    print(f"  Aggregate dir: {aggregate_dir}")
    print()
    
    # Scan and examine files
    scan_aggregate_directory(aggregate_dir, max_files=3)
    
    print()
    print("=" * 80)
    print("EXAMINATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()