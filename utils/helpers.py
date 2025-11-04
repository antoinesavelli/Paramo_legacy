"""General utility helper functions."""

import hashlib
import json
from typing import Any, Dict, Optional
import pickle
import logging
from datetime import datetime
import pytz
import os


def calculate_hash(data: Any) -> str:
    """Calculate hash of data for caching."""
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    return hashlib.md5(data_str.encode()).hexdigest()


def validate_config(config: Any) -> bool:
    """Validate configuration has all required attributes."""
    try:
        # Basic nested existence checks
        required = [
            config.api.ALPACA_API_KEY,
            config.api.ALPACA_SECRET_KEY,
            config.risk.MAX_RISK_PER_TRADE,
            config.risk.MAX_DAILY_LOSS,
            config.risk.MAX_CONCURRENT_POSITIONS,
            config.screening.MIN_GAP_PERCENT
        ]
        if any(v is None for v in required):
            return False
        if config.risk.MAX_RISK_PER_TRADE <= 0 or config.risk.MAX_CONCURRENT_POSITIONS <= 0:
            return False
        return True
    except AttributeError:
        return False


def format_currency(amount: float) -> str:
    """Format number as currency."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format number as percentage."""
    return f"{value:.2f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safe division with default value."""
    return numerator / denominator if denominator != 0 else default


def calculate_compound_return(initial: float, final: float, periods: int) -> float:
    """Calculate compound annual growth rate."""
    if initial <= 0 or periods <= 0:
        return 0
    return ((final / initial) ** (1 / periods) - 1) * 100


def is_market_hours(timestamp: datetime, config: Any) -> bool:
    """Check if timestamp is during market hours."""
    eastern = pytz.timezone('US/Eastern')
    dt = timestamp.astimezone(eastern)
    if dt.weekday() >= 5:
        return False
    market_time = dt.time()
    return config.market_hours.PRE_MARKET_START <= market_time <= config.market_hours.AFTER_HOURS_END


def save_state(data: Dict, filepath: str) -> bool:
    """Save system state to file."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            bytes_written = f.tell() if hasattr(f, "tell") else "n/a"
        logging.info("State saved successfully path=%s bytes=%s hash=%s",
                     filepath, bytes_written, calculate_hash(data))
        return True
    except Exception as e:
        logging.error("Error saving state path=%s err=%s", filepath, e)
        return False


def load_state(filepath: str) -> Optional[Dict]:
    """Load system state from file."""
    try:
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        logging.info("State loaded successfully path=%s hash=%s", filepath, calculate_hash(state))
        return state
    except Exception as e:
        logging.error("Error loading state path=%s err=%s", filepath, e)
        return None


def log_and_return(logger, message, return_value):
    """Log error message and return a value (for error handling patterns)."""
    logger.error(message)
    return return_value


def check_first_bar(data_dir, date_str, symbol, tz='US/Eastern'):
    """Check for first available bar instead of opening bar."""
    from data_handler.local import load_day_with_first_bar
    
    # Simple file existence check - if no file exists, just skip
    parquet_path = os.path.join(data_dir, f"{date_str}.parquet")
    if not os.path.exists(parquet_path):
        print(f"No data for {symbol} on {date_str}: File not found")
        return

    # Try loading the data
    df = load_day_with_first_bar(data_dir, date_str, symbol, tz)
    if df.empty:
        print(f"No data for {symbol} on {date_str}")
        return

    if 'window_start' in df.columns:
        first_bar_time = df['window_start'].iloc[0]
        print(f"First available bar for {symbol} on {date_str} at {first_bar_time}:")
        first_bar = df[df['timestamp'] == first_bar_time]
        print(first_bar)
    else:
        print(f"No window_start column found for {symbol} on {date_str}.")
        if not df.empty:
            print("First row available:")
            print(df.head(1))
        else:
            print("No bars available for this symbol on this date.")
