"""
Temporary script to fix zero low values in existing daily aggregate files.
Rebuilds aggregates from source intraday data, filtering out invalid prices.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_aggregate_lows(
    data_dir: Path = Path(r"D:\trading_data"),
    aggregate_dir: Path = Path(r"D:\trading_data\daily_aggregates")
):
    """
    Fix zero low values in existing aggregate files by rebuilding from source data.
    """
    
    if not aggregate_dir.exists():
        logger.error(f"Aggregate directory not found: {aggregate_dir}")
        return
    
    # Find all existing aggregate files
    aggregate_files = sorted(aggregate_dir.glob("*/*.parquet"))
    logger.info(f"Found {len(aggregate_files)} aggregate files to check/fix")
    
    total_checked = 0
    total_fixed = 0
    zero_lows_found = 0
    
    for agg_file in tqdm(aggregate_files, desc="Fixing aggregate files"):
        try:
            # Read existing aggregate
            agg_df = pd.read_parquet(agg_file)
            
            if agg_df.empty:
                continue
            
            total_checked += 1
            
            # Check for zero lows
            zero_low_count = (agg_df['low'] == 0).sum()
            
            if zero_low_count == 0:
                logger.debug(f"✓ {agg_file.name} - No zero lows found")
                continue
            
            zero_lows_found += zero_low_count
            logger.info(f"⚠ {agg_file.name} - Found {zero_low_count} zero lows, rebuilding...")
            
            # Extract year-month from filename (YYYYMM.parquet)
            year_month = agg_file.stem
            year = year_month[:4]
            month = year_month[4:6]
            
            # Find all intraday files for this month
            intraday_dir = data_dir / year / month
            
            if not intraday_dir.exists():
                logger.warning(f"Source data not found: {intraday_dir}")
                continue
            
            daily_files = sorted(intraday_dir.glob("*.parquet"))
            
            if not daily_files:
                logger.warning(f"No daily files found in {intraday_dir}")
                continue
            
            # Rebuild aggregates for this month
            month_aggregates = []
            
            for file in daily_files:
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
                    
                    # ✅ FILTER OUT INVALID PRICES
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
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                        'timestamp': ['first', 'last', 'count']
                    }).reset_index()
                    
                    # Flatten multi-level columns
                    daily_stats.columns = [
                        'symbol', 'open', 'high', 'low', 'close', 'volume',
                        'first_timestamp', 'last_timestamp', 'bar_count'
                    ]
                    
                    daily_stats['date'] = date
                    month_aggregates.append(daily_stats)
                    
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {e}")
                    continue
            
            # Rebuild and save
            if month_aggregates:
                new_agg_df = pd.concat(month_aggregates, ignore_index=True)
                
                # Reorder columns
                new_agg_df = new_agg_df[[
                    'date', 'symbol', 'open', 'high', 'low', 'close', 
                    'volume', 'bar_count', 'first_timestamp', 'last_timestamp'
                ]]
                
                new_agg_df = new_agg_df.sort_values(['date', 'symbol'])
                
                # Save back to file
                new_agg_df.to_parquet(
                    agg_file,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )
                
                # Verify fix
                new_zero_count = (new_agg_df['low'] == 0).sum()
                logger.info(
                    f"✓ Fixed {agg_file.name}: {zero_low_count} → {new_zero_count} zero lows "
                    f"({len(new_agg_df)} records)"
                )
                total_fixed += 1
            else:
                logger.warning(f"No valid data found to rebuild {agg_file.name}")
                
        except Exception as e:
            logger.error(f"Error fixing {agg_file.name}: {e}")
            continue
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("REPAIR SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files checked: {total_checked}")
    logger.info(f"Files with zero lows: {total_fixed}")
    logger.info(f"Total zero lows found: {zero_lows_found}")
    logger.info("=" * 80)

if __name__ == "__main__":
    fix_aggregate_lows()