"""General utility helper functions."""

import hashlib
import json
from typing import Any, Dict, Optional
import logging
from datetime import datetime
import pytz
import os
import pandas as pd
import numpy as np


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


def validate_ohlcv(
    df: pd.DataFrame,
    source: str = "",
    logger: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Drop rows with clearly invalid OHLCV data without altering trading logic.

    Only removes records that would produce undefined behaviour downstream:
    negative prices, NaN / infinite core fields, or high < low.

    Args:
        df:      DataFrame expected to contain open, high, low, close, volume.
        source:  Label used in log messages (e.g. symbol + date for context).
        logger:  Optional logger; falls back to the module-level logging root.

    Returns:
        Cleaned DataFrame with the same schema.  May be empty if every row is
        invalid.  Never raises.
    """
    _log = logger if logger is not None else logging.getLogger("data_handler.ohlcv_validator")

    if df is None or df.empty:
        return df

    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    vol_cols   = [c for c in ("volume",) if c in df.columns]
    core_cols  = price_cols + vol_cols

    original_len = len(df)
    mask = pd.Series(True, index=df.index)

    # 1. Drop rows where any core field is NaN or infinite.
    for col in core_cols:
        mask &= df[col].notna() & np.isfinite(df[col].values)

    # 2. Drop rows with non-positive prices.
    for col in price_cols:
        mask &= df[col] > 0

    # 3. Drop rows where high < low (fundamental OHLCV integrity violation).
    if "high" in df.columns and "low" in df.columns:
        mask &= df["high"] >= df["low"]

    dropped = int(original_len - mask.sum())
    if dropped > 0:
        _log.warning(
            "validate_ohlcv[%s]: dropped %d/%d rows with invalid OHLCV data",
            source, dropped, original_len,
        )
        return df[mask].reset_index(drop=True)

    return df


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


def log_api_error(logger, msg: str, exc: Exception) -> None:
    """Log an API error with standard format. Centralizes error logging for API calls."""
    logger.error("%s: %s", msg, exc)


def log_db_error(logger, msg: str, exc: Exception) -> None:
    """Log a database error with standard format. Centralizes error logging for DB calls."""
    logger.error("%s: %s", msg, exc)


def calc_gap_percent(price: float, prev_close: float) -> float:
    """Calculate gap percentage from previous close. Canonical formula shared across modules."""
    return ((price - prev_close) / prev_close) * 100.0


def fetch_split_symbols(api, date: datetime) -> set:
    """
    Return the set of symbols that had a stock split effective on *date*.

    Uses Alpaca's corporate_actions announcements endpoint (type=split).
    Returns an empty set on any error so the screener never hard-fails.

    Args:
        api: Alpaca REST API client (tradeapi.REST instance)
        date: The date to check for splits (only the date portion is used)
    """
    try:
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        # Alpaca corporate_actions endpoint: GET /v2/corporate_actions/announcements
        # ca_types: R = reverse split, SS = stock split, SO = spinoff
        params = {
            'ca_types': 'R,SS',         # reverse splits + forward splits
            'since': date_str,
            'until': date_str,
        }
        announcements = _call_with_timeout_or_direct(api, params)
        symbols = set()
        for ann in announcements or []:
            sym = getattr(ann, 'symbol', None) or (ann.get('symbol') if isinstance(ann, dict) else None)
            if sym:
                symbols.add(str(sym).upper())
        return symbols
    except Exception:
        # Never let split-checking crash the screener
        return set()


def _call_with_timeout_or_direct(api, params: dict):
    """Helper: call Alpaca announcements endpoint, handling both SDK and REST variants."""
    import concurrent.futures
    try:
        # alpaca-trade-api SDK v2+: get_corporate_announcements
        def _call():
            return api.get_corporate_announcements(**params)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_call).result(timeout=10.0)
    except AttributeError:
        # Older SDK doesn't have get_corporate_announcements — treat as no splits
        return []
    except Exception:
        return []
