"""
Build daily aggregate cache from intraday parquet files.
Creates D:\trading_data\daily_aggregates\YYYY\YYYYMM.parquet

Schema:
- date, symbol, open, high, low, close, volume, bar_count
- Indexed on (symbol, date) for fast lookups
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_rolling_volume_averages(symbol_data: pd.DataFrame, lookback_periods: List[int] = [10, 20]) -> pd.DataFrame:
    """
    Calculate rolling volume averages for a single symbol.
    
    Args:
        symbol_data: DataFrame with 'date' and 'volume' columns sorted by date
        lookback_periods: List of periods to calculate (e.g., [10, 20])
        
    Returns:
        DataFrame with additional avg_volume_Xd columns
    """
    df = symbol_data.copy()
    
    for period in lookback_periods:
        col_name = f'avg_volume_{period}d'
        df[col_name] = df['volume'].rolling(window=period, min_periods=1).mean()
    
    return df

def build_daily_aggregates(
    data_dir: Path = Path(r"D:\trading_data"),
    output_dir: Path = Path(r"D:\trading_data\daily_aggregates")
):
    """
    Build daily aggregate cache from intraday files.
    
    File structure:
        Input:  D:\trading_data\YYYY\MM\YYYYMMDD.parquet (1-min bars, all symbols)
        Output: D:\trading_data\daily_aggregates\YYYY\YYYYMM.parquet (daily stats, all symbols)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all daily files
    daily_files = sorted(data_dir.glob("*/*/202*.parquet"))
    logger.info(f"Found {len(daily_files)} daily files to process")
    
    # Group files by year-month for efficient processing
    files_by_month = {}
    for file in daily_files:
        try:
            # Extract date from filename (YYYYMMDD.parquet)
            date_str = file.stem  # '20240103'
            year = date_str[:4]
            month = date_str[4:6]
            year_month = f"{year}{month}"
            
            if year_month not in files_by_month:
                files_by_month[year_month] = []
            files_by_month[year_month].append(file)
        except Exception as e:
            logger.warning(f"Skipping {file.name}: {e}")
    
    logger.info(f"Processing {len(files_by_month)} months")
    
    # Process each month
    for year_month, files in tqdm(files_by_month.items(), desc="Months"):
        year = year_month[:4]
        
        # Create output directory
        month_output_dir = output_dir / year
        month_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = month_output_dir / f"{year_month}.parquet"
        
        # Skip if already exists (for resuming)
        if output_file.exists():
            logger.info(f"Skipping {year_month} (already exists)")
            continue
        
        # Process all days in this month
        month_aggregates = []
        
        for file in tqdm(files, desc=f"{year_month}", leave=False):
            try:
                # Parse date from filename
                date_str = file.stem
                date = datetime.strptime(date_str, '%Y%m%d').date()
                
                # Read intraday file
                df = pd.read_parquet(file)
                
                if df.empty or 'symbol' not in df.columns:
                    continue
                
                # Ensure timestamp is datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                
                # ✅ FILTER OUT INVALID PRICES (zero or negative values)
                # This prevents zero values from being picked up in the min() aggregation
                df = df[
                    (df['open'] > 0) & 
                    (df['high'] > 0) & 
                    (df['low'] > 0) & 
                    (df['close'] > 0)
                ].copy()
                
                if df.empty:
                    continue
                
                # Group by symbol and aggregate
                daily_stats = df.groupby('symbol').agg({
                    'open': 'first',      # First bar open
                    'high': 'max',         # Day high
                    'low': 'min',          # Day low
                    'close': 'last',       # Last bar close
                    'volume': 'sum',       # Total volume
                    'timestamp': ['first', 'last', 'count']  # First/last timestamps, bar count
                }).reset_index()
                
                # Flatten multi-level columns
                daily_stats.columns = [
                    'symbol', 'open', 'high', 'low', 'close', 'volume',
                    'first_timestamp', 'last_timestamp', 'bar_count'
                ]
                
                # Add date column
                daily_stats['date'] = date
                
                month_aggregates.append(daily_stats)
                
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
                continue
        
        # Combine all days in month
        if month_aggregates:
            month_df = pd.concat(month_aggregates, ignore_index=True)
            
            # Reorder columns
            month_df = month_df[ [
                'date', 'symbol', 'open', 'high', 'low', 'close', 
                'volume', 'bar_count', 'first_timestamp', 'last_timestamp'
            ]]
            
            # Sort by date, symbol for efficient lookups
            month_df = month_df.sort_values(['date', 'symbol'])
            
            # Save as parquet with compression
            month_df.to_parquet(
                output_file,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            logger.info(
                f"✓ {year_month}: {len(month_df)} records "
                f"({len(month_df['symbol'].unique())} symbols, "
                f"{len(month_df['date'].unique())} days)"
            )
    
    logger.info("Daily aggregate build complete!")

if __name__ == "__main__":
    build_daily_aggregates()
