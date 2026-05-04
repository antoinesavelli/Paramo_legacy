"""
Build daily aggregate cache from intraday parquet files.
Creates S:\\trading\\daily_aggregates\\YYYY\\YYYY-MM.parquet

Schema:
- date, symbol, open, high, low, close, volume, bar_count
- float (shares outstanding), marketcap (float * open)
- Indexed on (symbol, date) for fast lookups

FILE FORMAT: YYYY-MM.parquet (ISO date format)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_rolling_volume_averages(symbol_data: pd.DataFrame, lookback_periods: List[int] = [10, 20]) -> pd.DataFrame:  # ✅ FIXED: Added type hint
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

def load_fundamentals_for_date(fundamentals_dir: Path, date: datetime) -> pd.DataFrame:
    """
    Load fundamentals data for a specific date.
    
    Args:
        fundamentals_dir: Base directory for fundamentals (S:\\trading\\fundamentals)
        date: Date object
        
    Returns:
        DataFrame with 'symbol' and 'float' columns
    """
    year = f"{date.year:04d}"
    month = f"{date.month:02d}"
    date_str = date.strftime('%Y-%m-%d')
    
    fundamentals_file = fundamentals_dir / year / month / f"{date_str}.parquet"
    
    if not fundamentals_file.exists():
        return pd.DataFrame()
    
    try:
        # Load only the sharesbas column
        df = pd.read_parquet(fundamentals_file, columns=['symbol', 'sharesbas'])
        
        # Rename sharesbas to float
        df = df.rename(columns={'sharesbas': 'float'})
        
        # Remove duplicates and invalid values
        df = df.dropna(subset=['float'])
        df = df[df['float'] > 0]  # Filter out zero or negative values
        df = df.drop_duplicates(subset=['symbol'], keep='last')
        
        return df
        
    except Exception as e:
        logger.warning(f"Error loading fundamentals for {date_str}: {e}")
        return pd.DataFrame()

def build_daily_aggregates(
    data_dir: Path = Path(r"S:\trading\ticker_data"),
    fundamentals_dir: Path = Path(r"S:\trading\fundamentals"),
    output_dir: Path = Path(r"S:\trading\daily_aggregates")
):
    """
    Build daily aggregate cache from intraday files and fundamentals.
    
    File structure:
        Input:  S:\\trading\\ticker_data\\YYYY\\MM\\YYYY-MM-DD.parquet (1-min bars, all symbols)
                S:\\trading\\fundamentals\\YYYY\\MM\\YYYY-MM-DD.parquet (fundamental data)
        Output: S:\\trading\\daily_aggregates\\YYYY\\YYYY-MM.parquet (daily stats, all symbols)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ✅ NEW FORMAT: Find all YYYY-MM-DD.parquet files
    daily_files = sorted(data_dir.glob("*/*/????-??-??.parquet"))
    logger.info(f"Found {len(daily_files)} daily files to process")
    
    # Group files by year-month for efficient processing
    files_by_month = {}
    for file in daily_files:
        try:
            # Extract date from filename (YYYY-MM-DD.parquet)
            date_str = file.stem  # '2024-01-03'
            year_month = date_str[:7]  # '2024-01' (keep hyphen)
            
            if year_month not in files_by_month:
                files_by_month[year_month] = []
            files_by_month[year_month].append(file)
        except Exception as e:
            logger.warning(f"Skipping {file.name}: {e}")
    
    logger.info(f"Processing {len(files_by_month)} months")
    
    # Process each month
    for year_month, files in tqdm(files_by_month.items(), desc="Months"):
        year = year_month[:4]  # '2024'
        
        # Create output directory
        month_output_dir = output_dir / year
        month_output_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ NEW: Use YYYY-MM.parquet format (with hyphen)
        output_file = month_output_dir / f"{year_month}.parquet"
        
        # Skip if already exists (for resuming)
        if output_file.exists():
            logger.info(f"Skipping {year_month} (already exists)")
            continue
        
        # Process all days in this month
        month_aggregates = []
        
        for file in tqdm(files, desc=f"{year_month}", leave=False):
            try:
                # Parse date from filename (YYYY-MM-DD)
                date_str = file.stem
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
                
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
                
                # ✅ Load and merge fundamentals data
                fundamentals = load_fundamentals_for_date(fundamentals_dir, datetime.combine(date, datetime.min.time()))  # ✅ FIXED: Convert date to datetime
                
                if not fundamentals.empty:
                    # Merge fundamentals with daily stats
                    daily_stats = daily_stats.merge(
                        fundamentals[['symbol', 'float']], 
                        on='symbol', 
                        how='left'
                    )
                    
                    # Calculate market cap = float * open
                    daily_stats['marketcap'] = daily_stats['float'] * daily_stats['open']
                else:
                    # Add empty columns if no fundamentals available
                    daily_stats['float'] = None
                    daily_stats['marketcap'] = None
                
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
                'volume', 'bar_count', 'float', 'marketcap',
                'first_timestamp', 'last_timestamp'
            ] ]
            
            # Sort by date, symbol for efficient lookups
            month_df = month_df.sort_values(['date', 'symbol'])
            
            # Save as parquet with compression
            month_df.to_parquet(
                output_file,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            # Log summary statistics
            symbols_with_fundamentals = month_df['float'].notna().sum()
            total_records = len(month_df)
            
            logger.info(
                f"✓ {year_month}: {total_records} records "
                f"({len(month_df['symbol'].unique())} symbols, "
                f"{len(month_df['date'].unique())} days, "
                f"{symbols_with_fundamentals}/{total_records} with fundamentals)"
            )
    
    logger.info("Daily aggregate build complete!")

if __name__ == "__main__":
    build_daily_aggregates()
