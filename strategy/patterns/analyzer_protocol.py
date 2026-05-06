# =====================================================
# analyzer_protocol.py - Shared analyzer interface
# =====================================================

from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PatternAnalyzerProtocol(Protocol):
    """
    Structural interface that PatternAnalyzer, ClaudePatternAnalyzer, and
    DualPatternAnalyzer all satisfy.  No inheritance required — any class
    with a matching ``analyze_pattern`` signature conforms automatically.
    """

    def analyze_pattern(
        self,
        symbol: str,
        bars: Optional[pd.DataFrame] = None,
        cache_key: Optional[str] = None,
        is_premarket: bool = False,
        gap_percent: Optional[float] = None,
    ) -> Dict: ...