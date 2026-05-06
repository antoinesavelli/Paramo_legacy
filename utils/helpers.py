"""General utility helper functions."""

import hashlib
import json
from typing import Any, Dict, Optional
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
    """Check if timestamp is during market hours (premarket open → after-hours close)."""
    from datetime import time as dt_time
    eastern = pytz.timezone('US/Eastern')
    dt = timestamp.astimezone(eastern)
    if dt.weekday() >= 5:
        return False
    market_time = dt.time()
    start = dt_time(*map(int, config.session.PREMARKET_START_ET.split(":")))
    end   = dt_time(*map(int, config.session.AFTER_HOURS_END_ET.split(":")))
    return start <= market_time <= end


def save_state(data: Dict, filepath: str) -> bool:
    """Persist system state to a JSON file.

    The data dict must contain only JSON-serialisable values (str, int, float,
    bool, None, list, dict).  Non-serialisable types (e.g. datetime) must be
    converted by the caller before passing them in.
    """
    try:
        path = os.path.abspath(filepath)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logging.info("State saved path=%s hash=%s", path, calculate_hash(data))
        return True
    except (TypeError, ValueError) as e:
        logging.error("State contains non-serialisable value path=%s err=%s", filepath, e)
        return False
    except OSError as e:
        logging.error("Error saving state path=%s err=%s", filepath, e)
        return False


def load_state(filepath: str) -> Optional[Dict]:
    """Load system state from a JSON file."""
    try:
        path = os.path.abspath(filepath)
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        logging.info("State loaded path=%s hash=%s", path, calculate_hash(state))
        return state
    except (json.JSONDecodeError, ValueError) as e:
        logging.error("State file corrupt or invalid JSON path=%s err=%s", filepath, e)
        return None
    except OSError as e:
        logging.error("Error loading state path=%s err=%s", filepath, e)
        return None


def log_and_return(logger, message, return_value):
    """Log error message and return a value (for error handling patterns)."""
    logger.error(message)
    return return_value


def check_first_bar(data_dir, date_str, symbol, tz='US/Eastern'):
    """Check for first available bar instead of opening bar."""
    from data_handler.local import load_day_with_first_bar

    parquet_path = os.path.join(data_dir, f"{date_str}.parquet")
    if not os.path.exists(parquet_path):
        logging.warning("check_first_bar: no data file symbol=%s date=%s path=%s",
                        symbol, date_str, parquet_path)
        return

    df = load_day_with_first_bar(data_dir, date_str, symbol, tz)
    if df.empty:
        logging.warning("check_first_bar: empty dataframe symbol=%s date=%s",
                        symbol, date_str)
