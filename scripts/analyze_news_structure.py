"""
Analyze News Parquet Files Structure
Inspects parquet files in D:\trading_data\news to understand schema and content
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def analyze_news_files():
    """Analyze structure and content of news parquet files."""
    
    news_dir = Path(r"D:\trading_data\news")
    
    print("=" * 80)
    print("NEWS PARQUET FILE STRUCTURE ANALYSIS")
    print("=" * 80)
    print(f"Directory: {news_dir}")
    print()
    
    # Check if directory exists
    if not news_dir.exists():
        print(f"ERROR: Directory does not exist: {news_dir}")
        return
    
    # Find all parquet files
    parquet_files = sorted(list(news_dir.glob("*.parquet")))
    
    if not parquet_files:
        print("ERROR: No parquet files found in directory")
        return
    
    print(f"Total parquet files found: {len(parquet_files)}")
    print()
    
    # Analyze date range
    dates = []
    for file in parquet_files:
        try:
            # Parse YYYYMMDD.parquet format
            date_str = file.stem  # Gets filename without extension
            if len(date_str) == 8 and date_str.isdigit():
                date = datetime.strptime(date_str, "%Y%m%d")
                dates.append(date)
        except:
            pass
    
    if dates:
        print(f"Date Range:")
        print(f"  First file: {min(dates).strftime('%Y-%m-%d')}")
        print(f"  Last file:  {max(dates).strftime('%Y-%m-%d')}")
        print(f"  Total days: {len(dates)}")
        print()
    
    # Analyze first 5 files in detail
    print("=" * 80)
    print("DETAILED ANALYSIS OF SAMPLE FILES")
    print("=" * 80)
    print()
    
    for i, file_path in enumerate(parquet_files[:5], 1):
        print(f"File {i}: {file_path.name}")
        print("-" * 80)
        
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            # Basic info
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print()
            
            # Column info
            print("Columns and Types:")
            for col in df.columns:
                dtype = df[col].dtype
                null_count = df[col].isna().sum()
                null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
                print(f"  • {col:25s} | {str(dtype):20s} | Nulls: {null_count:5d} ({null_pct:5.1f}%)")
            print()
            
            # Sample data (first 3 rows)
            if len(df) > 0:
                print("Sample Data (first 3 rows):")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', 50)
                print(df.head(3).to_string(index=False))
                print()
            
            # Statistics by column type
            print("Data Statistics:")
            
            # Numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
            if len(numeric_cols) > 0:
                print("\n  Numeric Columns:")
                for col in numeric_cols:
                    print(f"    {col}:")
                    print(f"      Min: {df[col].min()}")
                    print(f"      Max: {df[col].max()}")
                    print(f"      Mean: {df[col].mean():.2f}")
                    print(f"      Median: {df[col].median():.2f}")
            
            # String columns
            string_cols = df.select_dtypes(include=['object', 'string']).columns
            if len(string_cols) > 0:
                print("\n  String Columns:")
                for col in string_cols:
                    try:
                        unique_count = df[col].nunique()
                    except TypeError:
                        unique_count = len(df[col])  # Just count rows for list columns
                    
                    print(f"    {col}: {unique_count} unique values")
                    if unique_count <= 10:
                        print(f"      Values: {sorted(df[col].unique().tolist())}")
                    else:
                        top_5 = df[col].value_counts().head(5)
                        print(f"      Top 5 values:")
                        for val, count in top_5.items():
                            print(f"        '{val}': {count}")
            
            # Date/timestamp columns
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                print("\n  Date/Timestamp Columns:")
                for col in date_cols:
                    print(f"    {col}:")
                    print(f"      Min: {df[col].min()}")
                    print(f"      Max: {df[col].max()}")
                    print(f"      Timezone: {df[col].dt.tz if hasattr(df[col].dt, 'tz') else 'naive'}")
            
            # List columns (if any)
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if first non-null value is a list
                    first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if isinstance(first_val, list):
                        print(f"\n  List Column: {col}")
                        avg_len = df[col].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
                        print(f"    Average list length: {avg_len:.2f}")
                        print(f"    Sample values: {first_val[:3] if len(first_val) > 0 else 'empty'}")
            
            # Check for symbol distribution
            if 'symbol' in df.columns:
                print(f"\n  Symbol Distribution:")
                print(f"    Unique symbols: {df['symbol'].nunique()}")
                print(f"    Rows per symbol (avg): {len(df) / df['symbol'].nunique():.2f}")
                print(f"    Top 10 symbols by article count:")
                top_symbols = df['symbol'].value_counts().head(10)
                for sym, count in top_symbols.items():
                    print(f"      {sym}: {count} entries")
            
            print()
            print("=" * 80)
            print()
            
        except Exception as e:
            print(f"ERROR reading file: {e}")
            import traceback
            traceback.print_exc()
            print()
            print("=" * 80)
            print()
    
    # Cross-file analysis
    if len(parquet_files) > 1:
        print("=" * 80)
        print("CROSS-FILE CONSISTENCY CHECK")
        print("=" * 80)
        print()
        
        # Check schema consistency across first 10 files
        schemas = []
        for file_path in parquet_files[:10]:
            try:
                df = pd.read_parquet(file_path)
                schema = {
                    'columns': set(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
                schemas.append((file_path.name, schema))
            except:
                pass
        
        if len(schemas) > 1:
            # Check if all schemas are identical
            first_cols = schemas[0][1]['columns']
            all_same = all(s[1]['columns'] == first_cols for s in schemas)
            
            if all_same:
                print("✓ All files have identical column sets")
            else:
                print("✗ Files have different columns:")
                for fname, schema in schemas:
                    print(f"  {fname}: {sorted(schema['columns'])}")
            
            print()
            
            # Check dtype consistency
            dtype_issues = []
            for col in first_cols:
                dtypes_for_col = [s[1]['dtypes'].get(col, 'missing') for s in schemas]
                if len(set(dtypes_for_col)) > 1:
                    dtype_issues.append((col, dtypes_for_col))
            
            if dtype_issues:
                print("✗ Dtype inconsistencies found:")
                for col, dtypes in dtype_issues:
                    print(f"  {col}: {set(dtypes)}")
            else:
                print("✓ All columns have consistent dtypes across files")
        
        print()
    
    # Summary recommendations
    print("=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    # Analyze first file for recommendations
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
        
        print("Expected Schema for NewsIntegrationBacktest:")
        print("  Required columns:")
        print("    • symbol (string) - Stock ticker")
        print("    • date (date32) - Date of news")
        print("    • article_count (int64) - Number of articles")
        print("    • avg_sentiment (float64) - Average sentiment [-1, 1]")
        print("    • avg_match_score (float64) - Average match score [0, 100]")
        print()
        
        print("  Optional columns:")
        print("    • total_highlights (int64)")
        print("    • source_domains (list<string>)")
        print("    • top_headlines (list<string>)")
        print("    • created_at (timestamp[ns])")
        print("    • catalyst_type (string)")
        print()
        
        # Check which columns are present
        required_cols = ['symbol', 'date', 'article_count', 'avg_sentiment', 'avg_match_score']
        optional_cols = ['total_highlights', 'source_domains', 'top_headlines', 'created_at', 'catalyst_type']
        
        print("Your file has:")
        for col in required_cols:
            status = "✓" if col in df.columns else "✗"
            print(f"  {status} {col}")
        
        print()
        print("Optional columns present:")
        for col in optional_cols:
            if col in df.columns:
                print(f"  ✓ {col}")
        
        # Check for unexpected columns
        expected_all = set(required_cols + optional_cols)
        unexpected = set(df.columns) - expected_all
        if unexpected:
            print()
            print("Additional columns (not used by backtest):")
            for col in sorted(unexpected):
                print(f"  • {col}")

if __name__ == "__main__":
    try:
        analyze_news_files()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print()
        input("Press Enter to exit...")