# =====================================================
# data_handler/base.py - Structural contract for all data handlers.
#
# Uses typing.Protocol (PEP 544) so that APIDataHandler and
# LocalDataHandler satisfy this interface automatically via
# structural subtyping — no inheritance changes needed.
#
# Consumers should annotate with DataHandler instead of the
# concrete class so they are testable and swappable.
#
# Usage:
#   from data_handler.base import DataHandler
#
#   class MyConsumer:
#       def __init__(self, data: DataHandler): ...
# =====================================================

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, runtime_checkable

import pandas as pd
from typing import Protocol


@runtime_checkable
class DataHandler(Protocol):
    """
    Structural interface shared by APIDataHandler (live) and
    LocalDataHandler (backtest).

    Any class that implements these methods satisfies this Protocol —
    no explicit subclassing required. The @runtime_checkable decorator
    allows isinstance(obj, DataHandler) checks in tests or guards.
    """

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_universe(self) -> pd.DataFrame:
        """
        Return the tradeable symbol universe.

        Returns:
            DataFrame with at minimum a 'symbol' column (str, uppercase).
        """
        ...

    # ------------------------------------------------------------------
    # Bar data
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return OHLCV bars for *symbol* within the given time range.

        Args:
            symbol:    Ticker symbol (e.g. 'AAPL').
            timeframe: Bar resolution string (e.g. '1Min', '1Day').
            start:     Inclusive start datetime (timezone-aware or naive ET).
            end:       Inclusive end datetime.
            limit:     If provided, return only the last *limit* bars.

        Returns:
            DataFrame with columns: symbol, timestamp, open, high, low,
            close, volume.  Empty DataFrame on failure — never raises.
        """
        ...

    def get_intraday_bars(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Min",
    ) -> pd.DataFrame:
        """
        Convenience wrapper around get_bars scoped to intraday resolution.

        Returns:
            Same contract as get_bars.
        """
        ...

    # ------------------------------------------------------------------
    # Availability check  (LocalDataHandler has this; APIDataHandler
    # can delegate to a trivial True return for live mode)
    # ------------------------------------------------------------------

    def has_data_for_date(self, date_str: str) -> bool:
        """
        Fast availability check for a calendar date.

        Args:
            date_str: ISO date string 'YYYY-MM-DD'.

        Returns:
            True if at least one symbol has data for this date.
        """
        ...