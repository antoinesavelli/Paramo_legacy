"""
Examine intraday parquet data structure and integrity.
Analyzes file structure, schema, data quality, and potential issues.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Date range for analysis - CONFIGURE AS NEEDED
START_DATE = "20241201"  # YYYYMMDD format
END_DATE = "20241205"    # YYYYMMDD format (inclusive)
DATA_DIR = Path(r"D:\trading_data")


def examine_intraday_file(
    file_path: Path,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive examination of a single intraday parquet file.
    
    Args:
        file_path: Path to parquet file
        detailed: Whether to perform detailed analysis
        
    Returns:
        Dict with examination results
    """
    results = {
        'file_path': str(file_path),
        'file_name': file_path.name,
        'file_size_mb': file_path.stat().st_size / (1024 * 1024),
        'exists': file_path.exists(),
        'readable': False,
        'error': None,
        'schema': {},
        'row_count': 0,
        'symbol_count': 0,
        'symbols': [],
        'columns': [],
        'dtypes': {},
        'data_quality': {},
        'price_statistics': {},
        'volume_statistics': {},
        'timestamp_analysis': {},
        'integrity_issues': []
    }
    
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        results['readable'] = True
        results['row_count'] = len(df)
        results['columns'] = list(df.columns)
        results['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Empty file check
        if df.empty:
            results['integrity_issues'].append("File is empty - no data rows")
            return results
        
        # Schema analysis
        results['schema'] = {
            'has_symbol': 'symbol' in df.columns,
            'has_timestamp': 'timestamp' in df.columns,
            'has_ohlc': all(col in df.columns for col in ['open', 'high', 'low', 'close']),
            'has_volume': 'volume' in df.columns,
            'has_cumulative_volume': 'cumulative_volume' in df.columns,
            'has_vwap': 'vwap' in df.columns,
            'has_count': 'count' in df.columns,
            'has_ms_of_day': 'ms_of_day' in df.columns,
            'has_date': 'date' in df.columns,
        }
        
        # Symbol analysis
        if 'symbol' in df.columns:
            symbols = df['symbol'].dropna().unique()
            results['symbol_count'] = len(symbols)
            results['symbols'] = sorted([str(s) for s in symbols if pd.notna(s)])
            
            # Check for null symbols
            null_symbols = df['symbol'].isna().sum()
            if null_symbols > 0:
                results['integrity_issues'].append(
                    f"Found {null_symbols} rows with null symbols"
                )
        else:
            results['integrity_issues'].append("Missing required 'symbol' column")
        
        # Timestamp analysis
        if 'timestamp' in df.columns:
            df_ts = df.copy()
            df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'], errors='coerce')
            
            null_ts = df_ts['timestamp'].isna().sum()
            if null_ts > 0:
                results['integrity_issues'].append(
                    f"Found {null_ts} rows with invalid/null timestamps"
                )
            
            valid_ts = df_ts[df_ts['timestamp'].notna()]
            if not valid_ts.empty:
                results['timestamp_analysis'] = {
                    'min_timestamp': str(valid_ts['timestamp'].min()),
                    'max_timestamp': str(valid_ts['timestamp'].max()),
                    'timezone': str(valid_ts['timestamp'].dt.tz) if hasattr(valid_ts['timestamp'].dt, 'tz') else 'None',
                    'null_count': int(null_ts),
                    'duplicate_count': int(df_ts.duplicated(subset=['symbol', 'timestamp']).sum())
                }
                
                # Check if timestamps are sorted
                is_sorted = valid_ts.groupby('symbol')['timestamp'].apply(
                    lambda x: x.is_monotonic_increasing
                ).all()
                if not is_sorted:
                    results['integrity_issues'].append(
                        "Timestamps are not sorted within symbols"
                    )
        else:
            results['integrity_issues'].append("Missing required 'timestamp' column")
        
        # Price data analysis
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            price_cols = ['open', 'high', 'low', 'close']
            
            for col in price_cols:
                # Check for zeros
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    results['integrity_issues'].append(
                        f"Found {zero_count} zero values in '{col}'"
                    )
                
                # Check for negatives
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    results['integrity_issues'].append(
                        f"Found {negative_count} negative values in '{col}'"
                    )
                
                # Check for nulls
                null_count = df[col].isna().sum()
                if null_count > 0:
                    results['integrity_issues'].append(
                        f"Found {null_count} null values in '{col}'"
                    )
            
            # Valid price data statistics
            valid_prices = df[
                (df['open'] > 0) & 
                (df['high'] > 0) & 
                (df['low'] > 0) & 
                (df['close'] > 0)
            ]
            
            results['data_quality']['valid_price_rows'] = len(valid_prices)
            results['data_quality']['invalid_price_rows'] = len(df) - len(valid_prices)
            results['data_quality']['valid_price_percentage'] = (
                len(valid_prices) / len(df) * 100 if len(df) > 0 else 0
            )
            
            if not valid_prices.empty:
                results['price_statistics'] = {
                    'open': {
                        'min': float(valid_prices['open'].min()),
                        'max': float(valid_prices['open'].max()),
                        'mean': float(valid_prices['open'].mean()),
                        'median': float(valid_prices['open'].median())
                    },
                    'high': {
                        'min': float(valid_prices['high'].min()),
                        'max': float(valid_prices['high'].max()),
                        'mean': float(valid_prices['high'].mean()),
                        'median': float(valid_prices['high'].median())
                    },
                    'low': {
                        'min': float(valid_prices['low'].min()),
                        'max': float(valid_prices['low'].max()),
                        'mean': float(valid_prices['low'].mean()),
                        'median': float(valid_prices['low'].median())
                    },
                    'close': {
                        'min': float(valid_prices['close'].min()),
                        'max': float(valid_prices['close'].max()),
                        'mean': float(valid_prices['close'].mean()),
                        'median': float(valid_prices['close'].median())
                    }
                }
                
                # Check OHLC logic (high >= low, etc.)
                invalid_hl = (valid_prices['high'] < valid_prices['low']).sum()
                if invalid_hl > 0:
                    results['integrity_issues'].append(
                        f"Found {invalid_hl} rows where high < low"
                    )
                
                invalid_open = (
                    (valid_prices['open'] > valid_prices['high']) |
                    (valid_prices['open'] < valid_prices['low'])
                ).sum()
                if invalid_open > 0:
                    results['integrity_issues'].append(
                        f"Found {invalid_open} rows where open is outside high/low range"
                    )
                
                invalid_close = (
                    (valid_prices['close'] > valid_prices['high']) |
                    (valid_prices['close'] < valid_prices['low'])
                ).sum()
                if invalid_close > 0:
                    results['integrity_issues'].append(
                        f"Found {invalid_close} rows where close is outside high/low range"
                    )
        else:
            results['integrity_issues'].append("Missing one or more OHLC columns")
        
        # Volume analysis
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            negative_volume = (df['volume'] < 0).sum()
            null_volume = df['volume'].isna().sum()
            
            if zero_volume > 0:
                results['integrity_issues'].append(
                    f"Found {zero_volume} rows with zero volume"
                )
            if negative_volume > 0:
                results['integrity_issues'].append(
                    f"Found {negative_volume} rows with negative volume"
                )
            if null_volume > 0:
                results['integrity_issues'].append(
                    f"Found {null_volume} rows with null volume"
                )
            
            valid_volume = df[df['volume'] > 0]
            if not valid_volume.empty:
                results['volume_statistics'] = {
                    'min': int(valid_volume['volume'].min()),
                    'max': int(valid_volume['volume'].max()),
                    'mean': float(valid_volume['volume'].mean()),
                    'median': float(valid_volume['volume'].median()),
                    'total': int(valid_volume['volume'].sum()),
                    'zero_count': int(zero_volume),
                    'negative_count': int(negative_volume)
                }
        else:
            results['integrity_issues'].append("Missing 'volume' column")
        
        # Cumulative volume check
        if 'cumulative_volume' in df.columns:
            null_cumvol = df['cumulative_volume'].isna().sum()
            if null_cumvol > 0:
                results['data_quality']['null_cumulative_volume'] = int(null_cumvol)
            
            # Check if cumulative volume is monotonically increasing per symbol
            if 'symbol' in df.columns:
                non_monotonic = df.groupby('symbol')['cumulative_volume'].apply(
                    lambda x: not x.is_monotonic_increasing
                ).sum()
                if non_monotonic > 0:
                    results['integrity_issues'].append(
                        f"Found {non_monotonic} symbols with non-monotonic cumulative_volume"
                    )
        
        # Memory usage
        results['memory_usage_mb'] = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
    except Exception as e:
        results['error'] = str(e)
        results['integrity_issues'].append(f"Failed to read file: {e}")
        logger.error(f"Error examining {file_path}: {e}")
    
    return results


def print_examination_report(results: Dict[str, Any], summary_mode: bool = False):
    """
    Print a formatted examination report.
    
    Args:
        results: Examination results dict
        summary_mode: If True, print condensed summary instead of full report
    """
    if summary_mode:
        # Condensed summary for multi-file analysis
        status = "✓" if results['readable'] and not results['integrity_issues'] else "✗"
        issue_count = len(results['integrity_issues'])
        logger.info(
            f"{status} {results['file_name']}: "
            f"{results['row_count']:,} rows, "
            f"{results['symbol_count']} symbols, "
            f"{issue_count} issues"
        )
        if issue_count > 0 and issue_count <= 3:
            for issue in results['integrity_issues']:
                logger.info(f"    - {issue}")
        elif issue_count > 3:
            logger.info(f"    - {results['integrity_issues'][0]}")
            logger.info(f"    - ... and {issue_count - 1} more issues")
        return
    
    # Full detailed report
    logger.info("")
    logger.info("=" * 80)
    logger.info("INTRADAY DATA EXAMINATION REPORT")
    logger.info("=" * 80)
    logger.info(f"File: {results['file_name']}")
    logger.info(f"Path: {results['file_path']}")
    logger.info(f"File Size: {results['file_size_mb']:.2f} MB")
    logger.info(f"Memory Usage: {results.get('memory_usage_mb', 0):.2f} MB")
    logger.info("")
    
    # File status
    logger.info("-" * 80)
    logger.info("FILE STATUS")
    logger.info("-" * 80)
    logger.info(f"Exists: {results['exists']}")
    logger.info(f"Readable: {results['readable']}")
    if results['error']:
        logger.error(f"Error: {results['error']}")
    logger.info("")
    
    if not results['readable']:
        return
    
    # Basic info
    logger.info("-" * 80)
    logger.info("BASIC INFORMATION")
    logger.info("-" * 80)
    logger.info(f"Total Rows: {results['row_count']:,}")
    logger.info(f"Total Symbols: {results['symbol_count']}")
    logger.info(f"Columns: {len(results['columns'])}")
    logger.info("")
    
    # Schema
    logger.info("-" * 80)
    logger.info("SCHEMA ANALYSIS")
    logger.info("-" * 80)
    for key, value in results['schema'].items():
        status = "✓" if value else "✗"
        logger.info(f"  {status} {key}: {value}")
    logger.info("")
    
    # Columns and types
    logger.info("-" * 80)
    logger.info("COLUMNS AND DATA TYPES")
    logger.info("-" * 80)
    for col, dtype in results['dtypes'].items():
        logger.info(f"  {col}: {dtype}")
    logger.info("")
    
    # Symbols (show first 20)
    if results['symbols']:
        logger.info("-" * 80)
        logger.info("SYMBOLS")
        logger.info("-" * 80)
        logger.info(f"Total unique symbols: {len(results['symbols'])}")
        display_symbols = results['symbols'][:20]
        logger.info(f"First 20: {', '.join(display_symbols)}")
        if len(results['symbols']) > 20:
            logger.info(f"... and {len(results['symbols']) - 20} more")
        logger.info("")
    
    # Timestamp analysis
    if results['timestamp_analysis']:
        logger.info("-" * 80)
        logger.info("TIMESTAMP ANALYSIS")
        logger.info("-" * 80)
        for key, value in results['timestamp_analysis'].items():
            logger.info(f"  {key}: {value}")
        logger.info("")
    
    # Data quality
    if results['data_quality']:
        logger.info("-" * 80)
        logger.info("DATA QUALITY")
        logger.info("-" * 80)
        for key, value in results['data_quality'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value:,}")
        logger.info("")
    
    # Price statistics
    if results['price_statistics']:
        logger.info("-" * 80)
        logger.info("PRICE STATISTICS (Valid Prices Only)")
        logger.info("-" * 80)
        for price_type, stats in results['price_statistics'].items():
            logger.info(f"  {price_type.upper()}:")
            for stat_name, value in stats.items():
                logger.info(f"    {stat_name}: ${value:,.2f}")
        logger.info("")
    
    # Volume statistics
    if results['volume_statistics']:
        logger.info("-" * 80)
        logger.info("VOLUME STATISTICS")
        logger.info("-" * 80)
        for key, value in results['volume_statistics'].items():
            if key in ['mean', 'median']:
                logger.info(f"  {key}: {value:,.2f}")
            else:
                logger.info(f"  {key}: {value:,}")
        logger.info("")
    
    # Integrity issues
    logger.info("-" * 80)
    logger.info("INTEGRITY ISSUES")
    logger.info("-" * 80)
    if results['integrity_issues']:
        for i, issue in enumerate(results['integrity_issues'], 1):
            logger.warning(f"  {i}. {issue}")
    else:
        logger.info("  ✓ No integrity issues found!")
    logger.info("")
    
    logger.info("=" * 80)


def print_aggregate_summary(all_results: List[Dict[str, Any]]):
    """Print aggregate summary across all examined files."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("AGGREGATE SUMMARY")
    logger.info("=" * 80)
    
    total_files = len(all_results)
    readable_files = sum(1 for r in all_results if r['readable'])
    files_with_issues = sum(1 for r in all_results if r['integrity_issues'])
    total_rows = sum(r['row_count'] for r in all_results)
    total_size_mb = sum(r['file_size_mb'] for r in all_results)
    
    # Collect all unique symbols across all files
    all_symbols = set()
    for r in all_results:
        all_symbols.update(r['symbols'])
    
    # Aggregate integrity issues
    issue_categories = {}
    for r in all_results:
        for issue in r['integrity_issues']:
            # Extract issue type (first part before details)
            issue_type = issue.split(':')[0] if ':' in issue else issue.split('Found')[0] if 'Found' in issue else issue
            issue_categories[issue_type] = issue_categories.get(issue_type, 0) + 1
    
    logger.info(f"Total Files Examined: {total_files}")
    logger.info(f"Readable Files: {readable_files}")
    logger.info(f"Files with Issues: {files_with_issues}")
    logger.info(f"Total Rows: {total_rows:,}")
    logger.info(f"Total Size: {total_size_mb:.2f} MB")
    logger.info(f"Unique Symbols: {len(all_symbols)}")
    logger.info("")
    
    if issue_categories:
        logger.info("-" * 80)
        logger.info("COMMON ISSUES")
        logger.info("-" * 80)
        for issue_type, count in sorted(issue_categories.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {count} files: {issue_type}")
    
    logger.info("=" * 80)


def generate_date_range(start_date: str, end_date: str) -> List[str]:
    """
    Generate list of dates between start and end (inclusive).
    
    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        
    Returns:
        List of date strings in YYYYMMDD format
    """
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    return dates


def main():
    """Main execution function."""
    logger.info("Starting intraday parquet data examination")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info("")
    
    # Generate date range
    try:
        dates = generate_date_range(START_DATE, END_DATE)
        logger.info(f"Examining {len(dates)} date(s)")
        logger.info("")
        
        all_results = []
        missing_files = []
        
        # Examine each date
        for date_str in dates:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            year = f"{date_obj.year}"
            month = f"{date_obj.month:02d}"
            filename = f"{date_str}.parquet"
            
            file_path = DATA_DIR / year / month / filename
            
            if not file_path.exists():
                missing_files.append(date_str)
                logger.warning(f"✗ {date_str}: File not found")
                continue
            
            # Examine the file
            results = examine_intraday_file(file_path, detailed=True)
            all_results.append(results)
            
            # Print report based on number of files
            if len(dates) == 1:
                # Single file - full detailed report
                print_examination_report(results, summary_mode=False)
            else:
                # Multiple files - summary mode
                print_examination_report(results, summary_mode=True)
        
        # Print aggregate summary for multi-file analysis
        if len(dates) > 1:
            print_aggregate_summary(all_results)
            
            if missing_files:
                logger.info("")
                logger.info(f"Missing files ({len(missing_files)}): {', '.join(missing_files)}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    main()