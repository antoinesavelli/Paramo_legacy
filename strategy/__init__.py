# =====================================================
# strategy/__init__.py - Trading Strategy Components
# =====================================================

"""
Trading strategy logic including pattern recognition and risk management.

This package contains the core strategy components that analyze market data
and make trading decisions. These components are independent of execution
mechanics and can be reused across live trading and backtesting.

Modules:
    pattern_analyzer: Pattern recognition and confluence scoring
    risk_manager: Position sizing and risk calculations
"""

from strategy.patterns.pattern_analyzer import PatternAnalyzer

# Import common risk management functions
from strategy.risk_manager import (
    calc_atr_stop,
    calc_position_size_percentage,
    calc_atr_trailing_stop
)

__all__ = [
    # Pattern Analysis
    'PatternAnalyzer',

    # Risk Management
    'calc_atr_stop',
    'calc_position_size_percentage',
    'calc_atr_trailing_stop',
]
